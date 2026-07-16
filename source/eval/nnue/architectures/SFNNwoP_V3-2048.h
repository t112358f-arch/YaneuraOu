// SFNNwoP_V3 : bucketごとに可変サイズのl1/l2を持つlayerstack architecture
// (nnue_arch_gen.pyにより自動生成)
//
// - ft_out はbucket間で共通 (2048)
// - l1/l2 はbucketごとに個別サイズ (--l1 / --l2 で指定)
// - bucket選択 (kingrank9 / progress8kpabs / progress9kpabs) はUSIオプション
//   LS_BUCKET_MODE で実行時に切り替える (architecture名には含めない)

#ifndef CLASSIC_NNUE_SFNNWOP_V3_2048_H_INCLUDED
#define CLASSIC_NNUE_SFNNWOP_V3_2048_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_ka_hm2.h"

#include <cstring>
#include <algorithm>
#include <string>

#include "../layers/affine_transform_explicit.h"
#include "../layers/affine_transform_sparse_input_explicit.h"
#include "../layers/clipped_relu_explicit.h"
#include "../layers/sqr_clipped_relu.h"

namespace YaneuraOu {
namespace Eval::NNUE {

using RawFeatures = Features::FeatureSet<
    Features::HalfKA_hm2<Features::Side::kFriend>>;

// 変換後の入力特徴量の次元数 (bucket間で共通)
constexpr IndexType kTransformedFeatureDimensions = 2048;

// NnueNetworks::network[] の要素数。SFNNwoP_V3は9bucket分をNetwork 1個に
// 集約するので常に1。実際のbucket数はkNumBuckets。
constexpr int LayerStacks = 1;
constexpr int kNumBuckets = 9;

constexpr IndexType kInputDims = kTransformedFeatureDimensions;

// bucketごとのL1/L2出力次元 (参考情報として公開)
constexpr IndexType kHidden1DimsPerBucket[kNumBuckets] = { 31,31,31,15,15,15,7,7,7 };
constexpr IndexType kHidden2DimsPerBucket[kNumBuckets] = { 96,96,96,32,32,32,32,32,32 };

// 1bucket分のネットワーク。L1/L2サイズをtemplate引数化することでbucketごとに
// 異なるサイズを持たせられる。
template <IndexType kHidden1, IndexType kHidden2, std::uint32_t kHash>
struct NetworkBucket {

    Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1 + 1> fc_0;
    Layers::ClippedReLUExplicit<kHidden1 + 1> ac_0;
    Layers::SqrClippedReLU<kHidden1 + 1> ac_sqr_0;

    Layers::AffineTransformExplicit<kHidden1 * 2, kHidden2> fc_1;
    Layers::ClippedReLUExplicit<kHidden2> ac_1;

    Layers::AffineTransformExplicit<kHidden2, 1> fc_2;

    using OutputType = std::int32_t;
    static constexpr IndexType kOutputDimensions = 1;

    static constexpr std::uint32_t GetHashValue() { return kHash; }

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

    struct alignas(kCacheLineSize) Buffer {
        alignas(kCacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
        alignas(kCacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
        alignas(kCacheLineSize) typename decltype(ac_sqr_0)::OutputType ac_sqr_0_out[CeilToMultiple<IndexType>(kHidden1 * 2, 32)];
        alignas(kCacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
        alignas(kCacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
        alignas(kCacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;
    };

    static constexpr std::size_t kBufferSize = sizeof(Buffer);

    const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, char* buffer) const {
        auto& buf = *reinterpret_cast<Buffer*>(buffer);

        fc_0.Propagate(transformedFeatures, buf.fc_0_out);
        ac_0.Propagate(buf.fc_0_out, buf.ac_0_out);
        ac_sqr_0.Propagate(buf.fc_0_out, buf.ac_sqr_0_out);
        std::memcpy(buf.ac_sqr_0_out + kHidden1, buf.ac_0_out,
            kHidden1 * sizeof(typename decltype(ac_0)::OutputType));
        fc_1.Propagate(buf.ac_sqr_0_out, buf.fc_1_out);
        ac_1.Propagate(buf.fc_1_out, buf.ac_1_out);
        fc_2.Propagate(buf.ac_1_out, buf.fc_2_out);

        buf.fc_2_out[0] += buf.fc_0_out[kHidden1];

        return buf.fc_2_out;
    }
};

using NetworkBucket0 = NetworkBucket<31, 96, 1704296506u>;
using NetworkBucket1 = NetworkBucket<31, 96, 1704296507u>;
using NetworkBucket2 = NetworkBucket<31, 96, 1704296508u>;
using NetworkBucket3 = NetworkBucket<15, 32, 3312284909u>;
using NetworkBucket4 = NetworkBucket<15, 32, 3312284910u>;
using NetworkBucket5 = NetworkBucket<15, 32, 3312284911u>;
using NetworkBucket6 = NetworkBucket<7, 32, 2505679272u>;
using NetworkBucket7 = NetworkBucket<7, 32, 2505679273u>;
using NetworkBucket8 = NetworkBucket<7, 32, 2505679274u>;

// 9bucket分の集約。NnueNetworksからは常にnetwork[0]の1個として扱われ、
// 実際のbucket選択はPropagate()の引数(0..kNumBuckets-1)で行う。
struct Network {

    NetworkBucket0 b0;
	NetworkBucket1 b1;
	NetworkBucket2 b2;
	NetworkBucket3 b3;
	NetworkBucket4 b4;
	NetworkBucket5 b5;
	NetworkBucket6 b6;
	NetworkBucket7 b7;
	NetworkBucket8 b8;

    using OutputType = std::int32_t;
    static constexpr IndexType kOutputDimensions = 1;

    static constexpr std::uint32_t GetHashValue() {
        return 1452418288u;
    }

    static std::string GetStructureString() {
        return "SFNNwoP-V3-2048-L1[31,31,31,15,15,15,7,7,7]-L2[96,96,96,32,32,32,32,32,32]";
    }

    Tools::Result ReadParameters(std::istream& stream) {
        bool ok = b0.ReadParameters(stream).is_ok()
			&& b1.ReadParameters(stream).is_ok()
			&& b2.ReadParameters(stream).is_ok()
			&& b3.ReadParameters(stream).is_ok()
			&& b4.ReadParameters(stream).is_ok()
			&& b5.ReadParameters(stream).is_ok()
			&& b6.ReadParameters(stream).is_ok()
			&& b7.ReadParameters(stream).is_ok()
			&& b8.ReadParameters(stream).is_ok();
        return ok ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
    }

    bool WriteParameters(std::ostream& stream) const {
        return b0.WriteParameters(stream)
			&& b1.WriteParameters(stream)
			&& b2.WriteParameters(stream)
			&& b3.WriteParameters(stream)
			&& b4.WriteParameters(stream)
			&& b5.WriteParameters(stream)
			&& b6.WriteParameters(stream)
			&& b7.WriteParameters(stream)
			&& b8.WriteParameters(stream);
    }

    static constexpr std::size_t kBufferSize = std::max({NetworkBucket0::kBufferSize, NetworkBucket1::kBufferSize, NetworkBucket2::kBufferSize, NetworkBucket3::kBufferSize, NetworkBucket4::kBufferSize, NetworkBucket5::kBufferSize, NetworkBucket6::kBufferSize, NetworkBucket7::kBufferSize, NetworkBucket8::kBufferSize});

    const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, char* buffer, int bucket) const {
        switch (bucket) {
        case 0: return b0.Propagate(transformedFeatures, buffer);
		case 1: return b1.Propagate(transformedFeatures, buffer);
		case 2: return b2.Propagate(transformedFeatures, buffer);
		case 3: return b3.Propagate(transformedFeatures, buffer);
		case 4: return b4.Propagate(transformedFeatures, buffer);
		case 5: return b5.Propagate(transformedFeatures, buffer);
		case 6: return b6.Propagate(transformedFeatures, buffer);
		case 7: return b7.Propagate(transformedFeatures, buffer);
		case 8: return b8.Propagate(transformedFeatures, buffer);
        default:
            return b0.Propagate(transformedFeatures, buffer);
        }
    }
};

}  // namespace Eval::NNUE
}  // namespace YaneuraOu

#endif // CLASSIC_NNUE_SFNNWOP_V3_2048_H_INCLUDED
