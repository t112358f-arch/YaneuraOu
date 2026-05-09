// header used in NNUE evaluation function
// NNUE評価関数で用いるheader

#ifndef CLASSIC_NNUE_EVALUATE_NNUE_H_INCLUDED
#define CLASSIC_NNUE_EVALUATE_NNUE_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_feature_transformer.h"
#include "nnue_architecture.h"
#include "../../misc.h"
#include "../../memory.h"
#include "../../shm.h"

#if defined(SFNNwoPSQT_KBT)
#include <memory>
#endif

namespace YaneuraOu {
namespace Eval::NNUE {

	#define EvalFileDefaultName "nn.bin"

	// Hash value of evaluation function structure
	// 評価関数の構造のハッシュ値
#if defined(SFNNwoPSQT)
	constexpr std::uint32_t kHashValue = 0x3c203b32u;
#if defined(SFNNwoPSQT_KBT)
	extern int kLayerStacks;
#else
	constexpr int kLayerStacks = LayerStacks;
#endif
#else
	constexpr std::uint32_t kHashValue =
	    FeatureTransformer::GetHashValue() ^ Network::GetHashValue();
	constexpr int kLayerStacks = 1;
#endif

	// NNUE評価関数パラメーターを格納する統合構造体。
	// 全メンバーが生配列で構成されており trivially copyable であるため、
	// プロセス間共有メモリに直接配置できる。
	struct NnueNetworks {
		FeatureTransformer feature_transformer;
#if defined(SFNNwoPSQT_KBT)
		LargePagePtr<Network[]> network;
#else
		Network network[kLayerStacks];
#endif
	};
#if defined(SFNNwoPSQT_KBT)
	struct FixedNnueNetworks {
		FeatureTransformer feature_transformer;
		Network network[LayerStacks];
	};
	static_assert(std::is_trivially_copyable_v<FixedNnueNetworks>,
		"FixedNnueNetworks must be trivially copyable for shared memory support");
#endif
#if !defined(SFNNwoPSQT_KBT)
	static_assert(std::is_trivially_copyable_v<NnueNetworks>,
		"NnueNetworks must be trivially copyable for shared memory support");
#endif

	// NNUE評価関数パラメーター（共有メモリまたはローカルメモリ上に配置）
#if defined(SFNNwoPSQT_KBT)
	extern LargePagePtr<NnueNetworks> network_storage;
	extern SystemWideSharedConstant<FixedNnueNetworks> shared_fixed_networks;
	extern bool use_fixed_network_storage;

	// KBTでは通常LayerStack数がjsonで決まるため動的配列を使う。
	// ただしLayerStacksと一致するときはV2と同じ固定長配列を共有メモリに置く。
	inline const FeatureTransformer& network_feature_transformer() {
		return use_fixed_network_storage ? (*shared_fixed_networks).feature_transformer
		                                 : network_storage->feature_transformer;
	}
	inline const Network& network_at(int index) {
		return use_fixed_network_storage ? (*shared_fixed_networks).network[index]
		                                 : network_storage->network[index];
	}
#else
	extern SystemWideSharedConstant<NnueNetworks> shared_networks;

	// 共有メモリ上のNnueNetworksへのconst参照を返すヘルパー。
	// 評価関数の呼び出しで毎回使われるので、インライン化する。
	inline const NnueNetworks& networks() { return *shared_networks; }
	inline const FeatureTransformer& network_feature_transformer() {
		return networks().feature_transformer;
	}
	inline const Network& network_at(int index) {
		return networks().network[index];
	}
#endif

	// 評価関数ファイル名
	extern const char* const kFileName;

	// 評価関数の構造を表す文字列を取得する
	std::string GetArchitectureString();

	// ヘッダを読み込む
	Tools::Result ReadHeader(std::istream& stream,
	    std::uint32_t* hash_value, std::string* architecture, std::uint32_t* version_out = nullptr);

	// ヘッダを書き込む
	bool WriteHeader(std::ostream& stream,
	    std::uint32_t hash_value, const std::string& architecture);

	// 評価関数パラメータを読み込む
	Tools::Result ReadParameters(std::istream& stream);

	// 評価関数パラメータを書き込む
	bool WriteParameters(std::ostream& stream);

} // namespace Eval::NNUE
} // namespace YaneuraOu

#if !defined(SFNNwoPSQT_KBT)
// NnueNetworks のコンテンツハッシュ。共有メモリの名前に使われる。
// 同一の評価関数パラメーターを持つプロセス同士で自動的にメモリが共有される。
template<>
struct std::hash<YaneuraOu::Eval::NNUE::NnueNetworks> {
	std::size_t operator()(const YaneuraOu::Eval::NNUE::NnueNetworks& n) const noexcept {
		return static_cast<std::size_t>(
			YaneuraOu::hash_bytes(reinterpret_cast<const char*>(&n), sizeof(n)));
	}
};
#else
template<>
struct std::hash<YaneuraOu::Eval::NNUE::FixedNnueNetworks> {
	std::size_t operator()(const YaneuraOu::Eval::NNUE::FixedNnueNetworks& n) const noexcept {
		return static_cast<std::size_t>(
			YaneuraOu::hash_bytes(reinterpret_cast<const char*>(&n), sizeof(n)));
	}
};
#endif

#endif  // defined(EVAL_NNUE)

#endif // #ifndef NNUE_EVALUATE_NNUE_H_INCLUDED
