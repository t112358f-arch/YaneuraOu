// Common SFNN without PSQT network implementation.
// Architecture headers should only select types and constants.

#ifndef CLASSIC_NNUE_SFNN_NETWORK_H_INCLUDED
#define CLASSIC_NNUE_SFNN_NETWORK_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"
#include "../layers/affine_transform_common_shard_input_explicit.h"
#include "../layers/affine_transform_explicit.h"
#include "../layers/affine_transform_sparse_input_explicit.h"
#include "../layers/clipped_relu_explicit.h"
#include "../layers/sqr_clipped_relu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

namespace YaneuraOu {
namespace Eval::NNUE {

// WSB (WithSharedBucket) 用: `Fc0Layer` (fc_0) が融合forward
// (`PropagatePair(input, other, outA, outB)`、同じ入力を2つのbucketの重みに
// 対してまとめて計算する) をサポートしているかどうかの検出。
// `AffineTransformSparseInputExplicit` は対応しているが、`sfnn_common_shard`
// 系アーキテクチャが使う `AffineTransformCommonShardInputExplicit` は
// (accumulator直読みという別種の複雑な経路のため) 今のところ対応していない
// — 未対応の場合は `SfnnNetwork::PropagatePair` が個別に `Propagate` を2回
// 呼ぶフォールバックへ自動的に切り替わる (常に正しく動く。融合による高速化
// だけが効かない)。
template <typename Fc0Layer, typename = void>
struct HasFusedFc0Pair : std::false_type {};

template <typename Fc0Layer>
struct HasFusedFc0Pair<Fc0Layer,
                        std::void_t<decltype(std::declval<const Fc0Layer&>().PropagatePair(
                            std::declval<const typename Fc0Layer::InputType*>(),
                            std::declval<const Fc0Layer&>(),
                            std::declval<typename Fc0Layer::OutputType*>(),
                            std::declval<typename Fc0Layer::OutputType*>()))>> : std::true_type {};

template <typename Fc0Layer, IndexType Hidden1Dims, IndexType Hidden2Dims, bool PackedTail>
struct SfnnNetworkBuffer;

template <typename Fc0Layer, IndexType Hidden1Dims, IndexType Hidden2Dims>
struct alignas(kCacheLineSize) SfnnNetworkBuffer<Fc0Layer, Hidden1Dims, Hidden2Dims, false> {
	alignas(kCacheLineSize) typename Fc0Layer::OutputBuffer fc_0_out;
	alignas(kCacheLineSize) typename Layers::SqrClippedReLU<Hidden1Dims>::OutputType
	    ac_sqr_0_out[CeilToMultiple<IndexType>(Hidden1Dims * 2, 32)];
	alignas(kCacheLineSize) typename Layers::AffineTransformExplicit<Hidden1Dims * 2, Hidden2Dims>::OutputBuffer fc_1_out;
	alignas(kCacheLineSize) typename Layers::ClippedReLUExplicit<Hidden2Dims>::OutputBuffer ac_1_out;
	alignas(kCacheLineSize) typename Layers::AffineTransformExplicit<Hidden2Dims, 1>::OutputBuffer fc_2_out;
};

template <typename Fc0Layer, IndexType Hidden1Dims, IndexType Hidden2Dims>
struct alignas(kCacheLineSize) SfnnNetworkBuffer<Fc0Layer, Hidden1Dims, Hidden2Dims, true> {
	alignas(kCacheLineSize) typename Fc0Layer::OutputBuffer fc_0_out;
	alignas(kCacheLineSize) typename Layers::AffineTransformExplicit<Hidden2Dims, 1>::OutputBuffer fc_2_out;
};

template <typename Fc0Layer, IndexType InputDims, IndexType Hidden1Dims, IndexType Hidden2Dims,
          bool UseShortcut = (Hidden1Dims % 8 == 7)>
struct SfnnNetwork {
	Fc0Layer fc_0;
	Layers::ClippedReLUExplicit<Hidden1Dims> ac_0;
	Layers::SqrClippedReLU<Hidden1Dims> ac_sqr_0;
	Layers::AffineTransformExplicit<Hidden1Dims * 2, Hidden2Dims> fc_1;
	Layers::ClippedReLUExplicit<Hidden2Dims> ac_1;
	Layers::AffineTransformExplicit<Hidden2Dims, 1> fc_2;

	using OutputType = std::int32_t;
	static constexpr IndexType kOutputDimensions = 1;
	static constexpr IndexType kInputDims = InputDims;
	static constexpr IndexType kHidden1Dims = Hidden1Dims;
	static constexpr IndexType kHidden2Dims = Hidden2Dims;
	static constexpr bool kUseShortcut = UseShortcut;
	static constexpr IndexType kHidden1OutputDims = kHidden1Dims + (kUseShortcut ? 1 : 0);

	static_assert(kHidden1Dims % 8 == 0 || kHidden1Dims % 8 == 7,
	              "SFNN H1 must be 8n without shortcut, or 8n-1 with shortcut.");
	static_assert(kUseShortcut == (kHidden1Dims % 8 == 7),
	              "SFNN shortcut is enabled only when H1 is 8n-1.");

#if defined(USE_AVX512)
	static constexpr bool kUsePackedTail = kHidden2Dims == 64;
#else
	static constexpr bool kUsePackedTail = false;
#endif

	using Buffer = SfnnNetworkBuffer<Fc0Layer, kHidden1Dims, kHidden2Dims, kUsePackedTail>;
	static constexpr std::size_t kBufferSize = sizeof(Buffer);

	static constexpr std::uint32_t GetHashValue() {
		return 0x6333718Au;
	}

	Tools::Result ReadParameters(std::istream& stream) {
		bool result = fc_0.ReadParameters(stream).is_ok()
			&& ac_0.ReadParameters(stream).is_ok()
			&& ac_sqr_0.ReadParameters(stream).is_ok()
			&& fc_1.ReadParameters(stream).is_ok()
			&& ac_1.ReadParameters(stream).is_ok()
			&& fc_2.ReadParameters(stream).is_ok();
		return result ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	bool WriteParameters(std::ostream& stream) const {
		return fc_0.WriteParameters(stream)
			&& ac_0.WriteParameters(stream)
			&& ac_sqr_0.WriteParameters(stream)
			&& fc_1.WriteParameters(stream)
			&& ac_1.WriteParameters(stream)
			&& fc_2.WriteParameters(stream);
	}

	static typename decltype(ac_0)::OutputType ClippedReLUValue(std::int32_t value) {
		const auto shifted = value >> kWeightScaleBits;
		if (shifted <= 0)
			return 0;
		if (shifted >= 127)
			return 127;
		return static_cast<typename decltype(ac_0)::OutputType>(shifted);
	}

	static typename decltype(ac_sqr_0)::OutputType SqrClippedReLUValue(std::int32_t value) {
		const auto sqr = (static_cast<long long>(value) * value) >> (2 * kWeightScaleBits + 7);
		return static_cast<typename decltype(ac_sqr_0)::OutputType>(sqr >= 127 ? 127 : sqr);
	}

	void MakeHidden1Input(const typename decltype(fc_0)::OutputType* input,
	                      typename decltype(ac_sqr_0)::OutputType* output) const {
		ac_sqr_0.PropagatePair(input, output, output + kHidden1Dims);
		std::fill(output + kHidden1Dims * 2,
		          output + CeilToMultiple<IndexType>(kHidden1Dims * 2, 32),
		          typename decltype(ac_sqr_0)::OutputType{0});
	}

	static void MakeHidden1InputPacked(const typename decltype(fc_0)::OutputType* input,
	                                   std::uint32_t* output) {
		constexpr IndexType kPackedWords = CeilToMultiple<IndexType>(kHidden1Dims * 2, 8) / 4;
		if constexpr (kHidden1Dims == 7) {
			const auto s0 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[0]));
			const auto s1 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[1]));
			const auto s2 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[2]));
			const auto s3 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[3]));
			const auto s4 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[4]));
			const auto s5 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[5]));
			const auto s6 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[6]));
			const auto c0 = static_cast<std::uint32_t>(ClippedReLUValue(input[0]));
			const auto c1 = static_cast<std::uint32_t>(ClippedReLUValue(input[1]));
			const auto c2 = static_cast<std::uint32_t>(ClippedReLUValue(input[2]));
			const auto c3 = static_cast<std::uint32_t>(ClippedReLUValue(input[3]));
			const auto c4 = static_cast<std::uint32_t>(ClippedReLUValue(input[4]));
			const auto c5 = static_cast<std::uint32_t>(ClippedReLUValue(input[5]));
			const auto c6 = static_cast<std::uint32_t>(ClippedReLUValue(input[6]));
			output[0] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
			output[1] = s4 | (s5 << 8) | (s6 << 16) | (c0 << 24);
			output[2] = c1 | (c2 << 8) | (c3 << 16) | (c4 << 24);
			output[3] = c5 | (c6 << 8);
		} else {
			std::memset(output, 0, kPackedWords * sizeof(std::uint32_t));
			auto bytes = reinterpret_cast<typename decltype(ac_sqr_0)::OutputType*>(output);
			for (IndexType i = 0; i < kHidden1Dims; ++i)
				bytes[i] = SqrClippedReLUValue(input[i]);
			for (IndexType i = 0; i < kHidden1Dims; ++i)
				bytes[kHidden1Dims + i] = ClippedReLUValue(input[i]);
		}
	}

	template <typename BufferType>
	const OutputType* PropagateTail(BufferType& buf) const {
		if constexpr (kUsePackedTail) {
			std::uint32_t ac_sqr_0_packed[CeilToMultiple<IndexType>(kHidden1Dims * 2, 8) / 4];
			MakeHidden1InputPacked(buf.fc_0_out, ac_sqr_0_packed);
			fc_1.PropagateClippedReLUPackedToOutput(ac_sqr_0_packed, fc_2, buf.fc_2_out);
		} else {
			MakeHidden1Input(buf.fc_0_out, buf.ac_sqr_0_out);
			fc_1.Propagate(buf.ac_sqr_0_out, buf.fc_1_out);
			ac_1.Propagate(buf.fc_1_out, buf.ac_1_out);
			fc_2.Propagate(buf.ac_1_out, buf.fc_2_out);
		}

		if constexpr (kUseShortcut)
			buf.fc_2_out[0] += buf.fc_0_out[kHidden1Dims];
		return buf.fc_2_out;
	}

	// WSB (WithSharedBucket) 用の出力ペア (`*this` = 選択bucket、`other` = 共有
	// bucket。呼び出し順は問わない、対称な演算)。
	struct PairOutput {
		OutputType a;
		OutputType b;
	};

	// [`PropagateTail`] のWSB用ペア版。`bufA`/`bufB` はそれぞれ `*this`/`other`
	// の fc_0 出力を含んだ状態で渡す (`fc_0.PropagatePair`/`PropagatePairFromAccumulator`
	// が事前に書き込む)。fc_1 以降は [`Layers::AffineTransformExplicit::PropagatePair`]
	// でまとめて1ループで計算する — ただし `kUsePackedTail` (H2==64のAVX512 packed
	// path) はすでに1回で完結する高度に最適化された経路なので、ここでは融合せず
	// 個別に呼ぶ (fc_0の融合だけで最大の行列積 (Hidden1入力) の重複計算/nnz列挙の
	// 重複を無くせるため、packed tail 自体を融合しなくても実質的な狙いは満たす)。
	template <typename BufferType>
	PairOutput PropagateTailPair(BufferType& bufA, const SfnnNetwork& other, BufferType& bufB) const {
		if constexpr (kUsePackedTail) {
			const OutputType* outA = PropagateTail(bufA);
			const OutputType* outB = other.PropagateTail(bufB);
			return {outA[0], outB[0]};
		} else {
			MakeHidden1Input(bufA.fc_0_out, bufA.ac_sqr_0_out);
			other.MakeHidden1Input(bufB.fc_0_out, bufB.ac_sqr_0_out);
			fc_1.PropagatePair(bufA.ac_sqr_0_out, bufB.ac_sqr_0_out, other.fc_1, bufA.fc_1_out, bufB.fc_1_out);
			ac_1.Propagate(bufA.fc_1_out, bufA.ac_1_out);
			other.ac_1.Propagate(bufB.fc_1_out, bufB.ac_1_out);
			fc_2.PropagatePair(bufA.ac_1_out, bufB.ac_1_out, other.fc_2, bufA.fc_2_out, bufB.fc_2_out);

			OutputType outA = bufA.fc_2_out[0];
			OutputType outB = bufB.fc_2_out[0];
			if constexpr (kUseShortcut) {
				outA += bufA.fc_0_out[kHidden1Dims];
				outB += bufB.fc_0_out[kHidden1Dims];
			}
			return {outA, outB};
		}
	}

	const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, char* buffer) const {
		auto& buf = *reinterpret_cast<Buffer*>(buffer);
		fc_0.Propagate(transformedFeatures, buf.fc_0_out);
		return PropagateTail(buf);
	}

	// WSB (WithSharedBucket) 用: `*this` (選択bucket) と `other` (共有bucket) の
	// forwardをまとめて1回で計算する。fc_0 (FT出力 -> Hidden1) は両bucketで
	// **入力 (`transformedFeatures`) が同じ**なので、1回のSIMDループで両方の
	// 重み行列に対する積和を行う ([`Layers::AffineTransformSparseInputExplicit::
	// PropagatePair`] — 疎入力のnnz列挙も1回で済む)。fc_1以降は
	// [`PropagateTailPair`] に委譲する。`bufferA`/`bufferB` は呼び出し側が別々に
	// 確保すること (`*this`用と`other`用、`PropagateTail`のように使い回すこと
	// はできない — 両方の中間活性を同時に保持する必要があるため)。
	//
	// 数値結果は `Propagate(transformedFeatures, bufferA)[0]` /
	// `other.Propagate(transformedFeatures, bufferB)[0]` を別々に呼んだ場合と
	// 完全に一致する。
	PairOutput PropagatePair(const TransformedFeatureType* transformedFeatures,
	                          const SfnnNetwork& other,
	                          char* bufferA,
	                          char* bufferB) const {
		auto& bufA = *reinterpret_cast<Buffer*>(bufferA);
		auto& bufB = *reinterpret_cast<Buffer*>(bufferB);
		if constexpr (HasFusedFc0Pair<Fc0Layer>::value) {
			fc_0.PropagatePair(transformedFeatures, other.fc_0, bufA.fc_0_out, bufB.fc_0_out);
		} else {
			// この `Fc0Layer` (例: `sfnn_common_shard` 系アーキテクチャの
			// `AffineTransformCommonShardInputExplicit`) は融合forwardに対応
			// していないので、個別に呼ぶ (正しさは保つ)。
			fc_0.Propagate(transformedFeatures, bufA.fc_0_out);
			other.fc_0.Propagate(transformedFeatures, bufB.fc_0_out);
		}
		return PropagateTailPair(bufA, other, bufB);
	}

#if defined(USE_AVX512)
	template <typename AccumulationType>
	const OutputType* PropagateFromAccumulator(const AccumulationType& accumulation,
	                                           Color sideToMove,
	                                           char* buffer) const {
		auto& buf = *reinterpret_cast<Buffer*>(buffer);
		fc_0.template PropagateSfnnFromAccumulator<kInputDims>(accumulation, sideToMove, buf.fc_0_out);
		return PropagateTail(buf);
	}

	// [`PropagatePair`] のAVX512 accumulator直読み版。fc_0 の
	// `PropagateSfnnFromAccumulator` 自体は (accumulatorのperspective走査/chunk
	// 処理という別種の複雑さを持つ特殊経路のため) 融合せず個別に呼ぶ — fc_1以降
	// ([`PropagateTailPair`]) はこのTransform経由版と共通の融合を使う。
	template <typename AccumulationType>
	PairOutput PropagatePairFromAccumulator(const AccumulationType& accumulation,
	                                        Color sideToMove,
	                                        const SfnnNetwork& other,
	                                        char* bufferA,
	                                        char* bufferB) const {
		auto& bufA = *reinterpret_cast<Buffer*>(bufferA);
		auto& bufB = *reinterpret_cast<Buffer*>(bufferB);
		fc_0.template PropagateSfnnFromAccumulator<kInputDims>(accumulation, sideToMove, bufA.fc_0_out);
		other.fc_0.template PropagateSfnnFromAccumulator<kInputDims>(accumulation, sideToMove, bufB.fc_0_out);
		return PropagateTailPair(bufA, other, bufB);
	}
#endif
};

}  // namespace Eval::NNUE
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif // CLASSIC_NNUE_SFNN_NETWORK_H_INCLUDED
