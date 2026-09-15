// SFNN without PSQT architecture

#ifndef CLASSIC_NNUE_SFNN_SFNN_HALFKAHM2_2296_15_64_ROUTERFT8FT8_H_INCLUDED
#define CLASSIC_NNUE_SFNN_SFNN_HALFKAHM2_2296_15_64_ROUTERFT8FT8_H_INCLUDED

#include "../features/feature_set.h"

#include "../features/half_ka_hm2.h"

#include "sfnn_network.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// Input features used in evaluation function
// 評価関数で用いる入力特徴量

    using RawFeatures = Features::FeatureSet<
        Features::HalfKA_hm2<Features::Side::kFriend>>;

    // Number of input feature dimensions after conversion
    // 変換後の入力特徴量の次元数
    constexpr IndexType kTransformedFeatureDimensions = 2304;

    // 小幅SFNN専用。従来幅のSFNNでは定義せず、既存の高速経路をそのまま使う。
    

    // Number of networks stored in the evaluation file
    constexpr int LayerStacks = 64;

    #define NNUE_SFNN_HAND_BUCKETS 1
    #define NNUE_SFNN_HAND_BUCKET_TYPE 0
    #define NNUE_SFNN_KING_BUCKETS 1
    #define NNUE_SFNN_KING_BUCKET_TYPE 0
    #define NNUE_SFNN_PROGRESS_BUCKETS 1
    
    #define NNUE_SFNN_ROUTER_MODE_NONE 0
    #define NNUE_SFNN_ROUTER_MODE_KPABS 1
    #define NNUE_SFNN_ROUTER_MODE_FTFT 2
    #define NNUE_SFNN_ROUTER_MODE 2
    // routerkpabsではバケット数そのもの、routerft<R>ft<R>ではRを表す (バケット数はR*R)。
    // 無印/router無しでは 0。
    #define NNUE_SFNN_ROUTER_N 8

    // `--bucket-mode ...routerft{R}ft{R}` (tatara) 用の net。FeatureTransformer の
    // accumulator に同居する router の生スコア数 (perspectiveごとに R 個)。
    // routerkpabs / router無しビルドでは定義しない (0扱い)。
    #define TANUKI_ROUTER_ARCH_FTBYFT
    constexpr IndexType kRouterFtByFtR = 8;


    // Number of groups for the first affine layer of SFNN.
    // common+shard fc_0でのみ2以上になる。
    constexpr IndexType kHidden1GroupCount = 1;

    // common+shard fc_0 settings. kHidden1ShardDimensions is per shard.
    constexpr bool kHidden1UsesCommonShard = false;
    constexpr IndexType kHidden1CommonDimensions = 0;
    constexpr IndexType kHidden1ShardDimensions = 0;

    // 各層の次元数
    // routerft<R>ft<R> のときは kTransformedFeatureDimensions (FT accumulator幅) より
    // R だけ狭い (router の生スコア分を除いた実際の特徴量次元)。それ以外は一致する。
    constexpr IndexType kInputDims   = 2296;
    constexpr IndexType kHidden1Dims = 15;
    constexpr bool kUseShortcut = true;
    constexpr IndexType kHidden1OutputDims = kHidden1Dims + (kUseShortcut ? 1 : 0);
    constexpr IndexType kHidden2Dims = 64;                              

    

    using Fc0Layer = Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1OutputDims>;
    using NetworkBase = SfnnNetwork<Fc0Layer, kInputDims, kHidden1Dims, kHidden2Dims, kUseShortcut>;

    struct Network : NetworkBase {
        static std::string GetStructureString() {
            return "SFNN_HALFKAHM2_2296_15_64_ROUTERFT8FT8";
        }
    };

}  // namespace Eval::NNUE
}  // namespace YaneuraOu

#endif // CLASSIC_NNUE_SFNN_HALFKAHM2_2296_15_64_ROUTERFT8FT8_H_INCLUDED
