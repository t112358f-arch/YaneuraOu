// SFNN without PSQT architecture

#ifndef CLASSIC_NNUE_SFNN_SFNN_HALFKAHM2_2304_15_64_PROGRESS8_WSB_ROUTERKPABS16_H_INCLUDED
#define CLASSIC_NNUE_SFNN_SFNN_HALFKAHM2_2304_15_64_PROGRESS8_WSB_ROUTERKPABS16_H_INCLUDED

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
    constexpr int LayerStacks = 129;

    #define NNUE_SFNN_HAND_BUCKETS 1
    #define NNUE_SFNN_HAND_BUCKET_TYPE 0
    #define NNUE_SFNN_KING_BUCKETS 1
    #define NNUE_SFNN_KING_BUCKET_TYPE 0
    #define NNUE_SFNN_PROGRESS_BUCKETS 8
    
    #define NNUE_SFNN_ROUTER_MODE_NONE 0
    #define NNUE_SFNN_ROUTER_MODE_KPABS 1
    #define NNUE_SFNN_ROUTER_MODE_FTFT 2
    #define NNUE_SFNN_ROUTER_MODE 1
    // routerkpabsではバケット数そのもの、routerft<R>ft<R>ではRを表す (バケット数はR*R)。
    // 無印/router無しでは 0。
    #define NNUE_SFNN_ROUTER_N 16


    // バケットindexの合成順序 (桁の重み、0=最上位桁 .. 3=最下位桁)。
    // `order<letters>` トークン省略時はデフォルト順序 hand(0)→king(1)→
    // progress(2)→router(3) (常に router が最下位桁になる、従来通りの順序)。
    #define NNUE_SFNN_ORDER_HAND 0
    #define NNUE_SFNN_ORDER_KING 1
    #define NNUE_SFNN_ORDER_PROGRESS 2
    #define NNUE_SFNN_ORDER_ROUTER 3

    // hand/king/progress/router の合成バケット数 (wsbの共有バケットを含まない)。
    #define NNUE_SFNN_SELECTABLE_BUCKETS 128

    // wsb (WithSharedBucket)。1のとき、「wsb の直前のカテゴリ以降 (合成順序上、
    // より下位の桁を全部含むブロック)」ごとに共有バケットが1個ずつ挿入され、
    // LayerStacks (=129) = (そのブロックより外側のバケット数の積) ×
    // (ブロックの選択バケット数 (=NNUE_SFNN_SHARED_BUCKET_INNER) + 1) になる。
    // 各共有バケットは、同じブロックに属する選択バケットのどれが選ばれても
    // 常に評価され、選択された1バケットとの出力平均が評価値になる
    // (`evaluate_nnue.cpp` の `ComputeScore` / `stack_index_for_nnue` 参照)。
    // 0なら従来通り選択された1バケットのみを使う。
    #define NNUE_SFNN_USE_SHARED_BUCKET 1
    // wsb有効時の「ブロック」のバケット数 (共有バケット自体の+1を含まない)。
    // wsb無効時は未使用 (NNUE_SFNN_SELECTABLE_BUCKETS と同じ値を入れてあるが
    // 参照されない)。
    #define NNUE_SFNN_SHARED_BUCKET_INNER 128

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
    constexpr IndexType kInputDims   = 2304;
    constexpr IndexType kHidden1Dims = 15;
    constexpr bool kUseShortcut = true;
    constexpr IndexType kHidden1OutputDims = kHidden1Dims + (kUseShortcut ? 1 : 0);
    constexpr IndexType kHidden2Dims = 64;                              

    #if !defined(ENABLE_SFNN_16BIT_WEIGHT)
    
    #if defined(NNUE_SFNN_FUSED_FT)
    #define NNUE_HAS_SFNN_ACCUMULATOR_PROPAGATE
    #define NNUE_HAS_SFNN16_ACCUMULATOR_PROPAGATE
    #endif
    
    #endif

    using Fc0Layer = Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1OutputDims>;
    #if defined(ENABLE_SFNN_16BIT_WEIGHT)
    using NetworkBase = SfnnNetwork16<kInputDims, kHidden1Dims, kHidden2Dims, kUseShortcut>;
    #else
    using NetworkBase = SfnnNetwork<Fc0Layer, kInputDims, kHidden1Dims, kHidden2Dims, kUseShortcut>;
    #endif

    struct Network : NetworkBase {
        static std::string GetStructureString() {
            return "SFNN_HALFKAHM2_2304_15_64_PROGRESS8_WSB_ROUTERKPABS16";
        }
    };

}  // namespace Eval::NNUE
}  // namespace YaneuraOu

#endif // CLASSIC_NNUE_SFNN_HALFKAHM2_2304_15_64_PROGRESS8_WSB_ROUTERKPABS16_H_INCLUDED
