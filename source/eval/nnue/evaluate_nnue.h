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

#if defined(SFNNwoPSQT)
#define NNUE_SFNN_KING_BUCKET_TYPE_NONE 0
#define NNUE_SFNN_KING_BUCKET_TYPE_K3K3 1
#define NNUE_SFNN_KING_BUCKET_TYPE_K9K9 2
#define NNUE_SFNN_KING_BUCKET_TYPE_K21K21 3
#define NNUE_SFNN_KING_BUCKET_TYPE_K29K29 4
#define NNUE_SFNN_KING_BUCKET_TYPE_K9K9Z 5
#define NNUE_SFNN_KING_BUCKET_TYPE_K13K13Z 6

#define NNUE_SFNN_HAND_BUCKET_TYPE_NONE 0
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND4 1
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND16 2
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND64 3
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND64Z 4
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND256 5
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND1024 6

#ifndef NNUE_SFNN_HAND_BUCKETS
#define NNUE_SFNN_HAND_BUCKETS 1
#endif
#ifndef NNUE_SFNN_HAND_BUCKET_TYPE
#if NNUE_SFNN_HAND_BUCKETS == 1
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_NONE
#elif NNUE_SFNN_HAND_BUCKETS == 4
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND4
#elif NNUE_SFNN_HAND_BUCKETS == 16
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND16
#elif NNUE_SFNN_HAND_BUCKETS == 64
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND64
#elif NNUE_SFNN_HAND_BUCKETS == 256
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND256
#elif NNUE_SFNN_HAND_BUCKETS == 1024
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND1024
#else
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_NONE
#endif
#endif
#ifndef NNUE_SFNN_KING_BUCKETS
#define NNUE_SFNN_KING_BUCKETS 9
#endif
#ifndef NNUE_SFNN_KING_BUCKET_TYPE
#if NNUE_SFNN_KING_BUCKETS == 9
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K3K3
#elif NNUE_SFNN_KING_BUCKETS == 81
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K9K9
#elif NNUE_SFNN_KING_BUCKETS == 169
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K13K13Z
#elif NNUE_SFNN_KING_BUCKETS == 441
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K21K21
#elif NNUE_SFNN_KING_BUCKETS == 841
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K29K29
#else
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_NONE
#endif
#endif
#ifndef NNUE_SFNN_PROGRESS_BUCKETS
#define NNUE_SFNN_PROGRESS_BUCKETS 1
#endif

// router バケット (tatara `--bucket-mode ...routerkpabs<N>` / `...routerft<R>ft<R>`)。
// routerkpabs / routerft<R>ft<R> は互いに排他 (同時使用不可)。詳細は
// architectures/nnue_arch_gen.py のコメント、および tatara側 router_kpabs.rs /
// router_ftbyft.rs を参照。
#define NNUE_SFNN_ROUTER_MODE_NONE 0
#define NNUE_SFNN_ROUTER_MODE_KPABS 1
#define NNUE_SFNN_ROUTER_MODE_FTFT 2
#ifndef NNUE_SFNN_ROUTER_MODE
#define NNUE_SFNN_ROUTER_MODE NNUE_SFNN_ROUTER_MODE_NONE
#endif
// routerkpabsではバケット数そのもの (N)、routerft<R>ft<R>ではR (バケット数はR*R)。
// router無しでは 0。
#ifndef NNUE_SFNN_ROUTER_N
#define NNUE_SFNN_ROUTER_N 0
#endif

// バケットindexの合成順序 (桁の重み、0=最上位桁 .. 3=最下位桁、`order<letters>`
// トークンで明示しない限りデフォルトの hand→king→progress→router、すなわち
// 常に router が最下位桁)。`stack_index_for_nnue` がこの順序で digits[] を
// 組み立てて合成する。
#ifndef NNUE_SFNN_ORDER_HAND
#define NNUE_SFNN_ORDER_HAND 0
#endif
#ifndef NNUE_SFNN_ORDER_KING
#define NNUE_SFNN_ORDER_KING 1
#endif
#ifndef NNUE_SFNN_ORDER_PROGRESS
#define NNUE_SFNN_ORDER_PROGRESS 2
#endif
#ifndef NNUE_SFNN_ORDER_ROUTER
#define NNUE_SFNN_ORDER_ROUTER 3
#endif

// hand/king/progress/router の合成バケット数 (wsbの共有バケットを含まない)。
#ifndef NNUE_SFNN_SELECTABLE_BUCKETS
#define NNUE_SFNN_SELECTABLE_BUCKETS 1
#endif

// wsb (WithSharedBucket、`--bucket-mode`/アーキ名のどこかに置く `wsb` トークン)。
// 1のとき、「wsb の直前のカテゴリ以降 (合成順序上、より下位の桁を全部含む
// ブロック)」ごとに、そのブロックの選択バケットに加えて「常に選ばれる共有
// バケット」を1個追加する。`LayerStacks` にはこの追加分が既に反映済み
// (`= outer_buckets * (NNUE_SFNN_SHARED_BUCKET_INNER + 1)`、outer_buckets は
// ブロックより外側のカテゴリの合成バケット数の積)。評価値は選択された1バケット
// と、同じブロックの共有バケットの出力平均になる (`evaluate_nnue.cpp` の
// `ComputeScore` / `stack_index_for_nnue` 参照)。0なら従来通り選択された
// 1バケットのみを使う。
//
// ⚠️ 破壊的変更: 旧仕様 (wsb は常にバケット名の末尾のみ許可、常に
// 「グローバルに1個だけの共有バケット」、共有バケットのindexは常に
// `kLayerStacks - 1`) から意味が変わっている。旧仕様の挙動を再現したい場合は
// wsb をバケット名の先頭に置く (`wsb_...`) こと。
#ifndef NNUE_SFNN_USE_SHARED_BUCKET
#define NNUE_SFNN_USE_SHARED_BUCKET 0
#endif
// wsb有効時の「ブロック」の選択バケット数 (共有バケット自体の+1を含まない)。
// wsb無効時は未使用 (`NNUE_SFNN_SELECTABLE_BUCKETS` と同じ値が入る想定だが
// 参照されない)。
#ifndef NNUE_SFNN_SHARED_BUCKET_INNER
#define NNUE_SFNN_SHARED_BUCKET_INNER 1
#endif
#endif

namespace YaneuraOu {
class Position;

namespace Eval::NNUE {

#if defined(ENABLE_SFNN_16BIT_WEIGHT)
#if !defined(SFNNwoPSQT)
#error ENABLE_SFNN_16BIT_WEIGHT requires an SFNN architecture
#endif
	#define EvalFileDefaultName "nn16.bin"
#else
	#define EvalFileDefaultName "nn.bin"
#endif

#if defined(SFNNwoPSQT) && NNUE_SFNN_PROGRESS_BUCKETS != 1
namespace Progress {

	// SFNNのLayerStack選択に使う進行度計算パラメーター。
	// EvalDir/progress.bin: headerless little-endian doubles, without bias.
	struct Parameters {
		static constexpr int kWeightCount = int(SQ_NB) * int(Eval::fe_end);

		Tools::Result ReadParameters(std::istream& stream);

		// 旧形式 (nn.bin内にProgressセクションを埋め込んでいた形式) の読み込み。
		// ストリームは、セクション先頭のhash(u32)の「次」にあること:
		//   int32 bias_q16 , int32 weights_q16[SQ_NB][fe_end]
		// 旧形式は進行度を0..255に量子化してからbucket化していたため、
		// bucket境界も旧形式のものを使う (bucket数が2の冪なら新形式と一致する)。
		static constexpr std::uint32_t kLegacySectionHash = 0x6f50524fu; // "oPRO"
		Tools::Result ReadLegacyParameters(std::istream& stream);

		std::int64_t SumQ16(const Position& pos) const;
		int BucketIndex(const Position& pos, int bucket_count) const;

		std::int32_t weights_q16_[SQ_NB][Eval::fe_end] = {};
		// 旧形式のみ非0 (新形式のprogress.binにbiasはない)。
		std::int32_t bias_q16_ = 0;
		// trueなら旧形式 (進行度0..255の量子化) のbucket境界を使う。
		bool legacy_thresholds_ = false;
	};

} // namespace Progress
#endif

#if defined(SFNNwoPSQT) && NNUE_SFNN_ROUTER_MODE == NNUE_SFNN_ROUTER_MODE_KPABS
namespace RouterKPAbs {

	// SFNNのLayerStack選択に使う、学習可能なバケット選択ネットワーク (tatara
	// `--bucket-mode ...routerkpabs<N>` / `router_kpabs.rs` 参照)。
	// progress8kpabs (Progress::Parameters) と全く同じ計算方法 (KP-absolute特徴の
	// 重み和、bias無し、Q16固定小数点) を N 出力 (N = NNUE_SFNN_ROUTER_N) に拡張したもの:
	//     logits[k] = Σ_i w[i][k]   (i は active な KP-absolute index、bias無し)
	//     bucket = argmax_k logits[k]
	// nn.bin内ではFeatureTransformerの直後 (Progress::Parametersがあればその後) に
	// このセクションを置く。
	struct Parameters {
		static constexpr int kN = NNUE_SFNN_ROUTER_N;

		// BucketIndex() の k (=bucket候補) 方向の総和をTARGET_CPUのSIMD幅で
		// 並列計算できるよう、出力方向をint32 SIMDレーン数の倍数に
		// パディングする。**ファイル上のレイアウトはこのパディング無し
		// (kN列のまま)** — ReadParameters/WriteParametersはkN列だけを
		// 読み書きする。パディング列 (kN..kNPadded-1) はメモリ上にしか
		// 存在せず常に0初期化 (BucketIndex()の最終argmaxはk<kNだけを見るので
		// 計算結果に影響しない)。こうしておくことで、異なるTARGET_CPU向けに
		// ビルドした実行ファイル間でも同じ .bin がそのまま読み込める
		// (SIMD幅はビルド依存だが、ファイル上の重み列数はビルド非依存)。
#if defined(USE_AVX512)
		using RouterVec = __m512i;
#elif defined(USE_AVX2)
		using RouterVec = __m256i;
#elif defined(USE_SSE2)
		using RouterVec = __m128i;
#elif defined(USE_NEON)
		using RouterVec = int32x4_t;
#else
		// SIMD命令が使えないTARGET_CPU向け。幅1 (=スカラー) にパディング
		// することで、後段のBucketIndex()を「幅1のSIMDループ」として
		// 実装した1本のコードで、SIMD版/スカラー版を分岐なく共有できる。
		using RouterVec = std::int32_t;
#endif
		static constexpr int kRouterSimdWidth = int(sizeof(RouterVec) / sizeof(std::int32_t));
		static constexpr int kNPadded = CeilToMultiple<int>(kN, kRouterSimdWidth);

		static constexpr std::uint32_t GetHashValue() {
			return 0x6f52544bu; // "oRTK" : NNUE router (kpabs) parameter section
		}

		Tools::Result ReadParameters(std::istream& stream);
		bool WriteParameters(std::ostream& stream) const;

		// side to move / non side to move それぞれのKP-absolute特徴 (progress8kpabsと
		// 同じピース列) の重み和を、bucket = 0..kN それぞれについて求め、argmaxを返す。
		int BucketIndex(const Position& pos) const;

		// 重み [sq][piece][bucket (0..kNPadded)] , bias無し。
		// [kN, kNPadded) は常に0 (上記コメント参照)。
		alignas(kCacheLineSize) std::int32_t weights_q16_[SQ_NB][Eval::fe_end][kNPadded] = {};
	};

} // namespace RouterKPAbs
#endif

	// Hash value of evaluation function structure
	// 評価関数の構造のハッシュ値
#if defined(SFNNwoPSQT)
	constexpr std::uint32_t kSfnnBaseHashValue = 0x3c203b32u;
#if NNUE_SFNN_ROUTER_MODE == NNUE_SFNN_ROUTER_MODE_KPABS
	constexpr std::uint32_t kRouterHashPart = RouterKPAbs::Parameters::GetHashValue();
#elif NNUE_SFNN_ROUTER_MODE == NNUE_SFNN_ROUTER_MODE_FTFT
	// routerft<R>ft<R> は専用のweightセクションを持たない (FeatureTransformerに同居する)
	// ため、代わりに R をhashに混ぜてarchitectureの取り違えを検知する。
	constexpr std::uint32_t kRouterHashPart = 0x6f465446u ^ static_cast<std::uint32_t>(NNUE_SFNN_ROUTER_N);
#else
	constexpr std::uint32_t kRouterHashPart = 0u;
#endif
#if NNUE_SFNN_USE_SHARED_BUCKET
	// wsb 有無で取り違えないよう hash に混ぜる (重み配列サイズ以外に構造上の差分が
	// 無いため、bucket数の一致だけでは wsb 有無を区別できない)。
	constexpr std::uint32_t kSharedBucketHashPart = 0x77534200u;
#else
	constexpr std::uint32_t kSharedBucketHashPart = 0u;
#endif
	// progress.binはnn.binとは別ファイルなので、progressはhashに含めない。
	constexpr std::uint32_t kHashValue =
	    kSfnnBaseHashValue ^ kRouterHashPart ^ kSharedBucketHashPart;
	constexpr int kLayerStacks = LayerStacks;
#if NNUE_SFNN_USE_SHARED_BUCKET
	// wsb有効時、「ブロック」(共有バケット1個+選択バケットNNUE_SFNN_SHARED_BUCKET_INNER個)
	// のサイズ。ブロックは kLayerStacks の中に outer_buckets 個並んでおり、
	// 各ブロックの末尾 (ブロック内の最後のindex) が、そのブロックに属する
	// bucketがどれ選ばれても常に評価される共有バケット。
	constexpr int kSharedBucketBlockSize = NNUE_SFNN_SHARED_BUCKET_INNER + 1;
	// `stack_index_for_nnue` が返す実際のbucket (選択側) から、同じブロックの
	// 共有バケットのindexを求める。旧仕様 (wsbは常にバケット名の末尾＝グローバルに
	// 1個だけの共有バケット、outer_buckets==1) では、どのbucketに対しても常に
	// 同じ値 (=kLayerStacks-1) を返す。
	constexpr int shared_bucket_index_for(int bucket) {
		const int outer_idx = bucket / kSharedBucketBlockSize;
		return outer_idx * kSharedBucketBlockSize + NNUE_SFNN_SHARED_BUCKET_INNER;
	}
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
#if defined(SFNNwoPSQT) && NNUE_SFNN_PROGRESS_BUCKETS != 1
		Progress::Parameters progress;
#endif
#if defined(SFNNwoPSQT) && NNUE_SFNN_ROUTER_MODE == NNUE_SFNN_ROUTER_MODE_KPABS
		// routerft<R>ft<R>は専用のweightを持たない (FeatureTransformerに同居するため)。
		RouterKPAbs::Parameters router_kpabs;
#endif
		Network network[kLayerStacks];
	};
	static_assert(std::is_trivially_copyable_v<NnueNetworks>,
		"NnueNetworks must be trivially copyable for shared memory support");

	// NNUE評価関数パラメーター（共有メモリまたはローカルメモリ上に配置）
	extern SystemWideSharedConstant<NnueNetworks> shared_networks;

	// 共有メモリ上のNnueNetworksへのconst参照を返すヘルパー。
	// 評価関数の呼び出しで毎回使われるので、インライン化する。
	inline const NnueNetworks& networks() { return *shared_networks; }

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
	// On a sidecar read failure, replace failed_file with its path.
	Tools::Result ReadParameters(std::istream& stream, std::string* failed_file = nullptr);

	// 評価関数パラメータを書き込む
	bool WriteParameters(std::ostream& stream);

} // namespace Eval::NNUE
} // namespace YaneuraOu

// NnueNetworks のコンテンツハッシュ。共有メモリの名前に使われる。
// 同一の評価関数パラメーターを持つプロセス同士で自動的にメモリが共有される。
template<>
struct std::hash<YaneuraOu::Eval::NNUE::NnueNetworks> {
	std::size_t operator()(const YaneuraOu::Eval::NNUE::NnueNetworks& n) const noexcept {
		return static_cast<std::size_t>(
			YaneuraOu::hash_bytes(reinterpret_cast<const char*>(&n), sizeof(n)));
	}
};

#endif  // defined(EVAL_NNUE)

#endif // #ifndef NNUE_EVALUATE_NNUE_H_INCLUDED
