# NNUE architecture header generator
#
#  NNUE評価関数のarchitecture headerを動的に生成するPythonで書かれたスクリプト。
# 

import argparse
import os
import re
import subprocess
import sys

def dedent4(text: str) -> str:
    # 各行の先頭4文字（スペース4つ）を削除して結合し直す
    # 行が4文字未満、あるいはスペースでない場合を考慮して lstrip でも可
    return "\n".join(line[4:] if line.startswith("    ") else line 
                        for line in text.strip("\n").splitlines())


print("NNUE architecture header generator by yaneurao V1.03 , 2026/07/20")

parser = argparse.ArgumentParser(description="NNUEのarchitecture headerを生成する。")
parser.add_argument('arch', type=str, nargs='?', default="halfkp_256x2-32-32", help="architectureを指定する。例) halfkp_1024x2-8-64, YANEURAOU_ENGINE_NNUE_HALFKP_1024X2_16_32とか")
parser.add_argument('out_dir', type=str, nargs='?', default=None, help="出力先のフォルダを指定する。省略時はこのスクリプトと同じフォルダ。")
parser.add_argument('--write-dummy-nn', type=str, default="", help="指定pathに、このarchitecture用のdummy nn.binを生成する。")
parser.add_argument('--dummy-mode', type=str, choices=("random-small", "zero"), default="random-small", help="dummy nn.binの初期化方式。デフォルトはrandom-small。")
parser.add_argument('--dummy-seed', type=int, default=20260722, help="random-small用の乱数seed。")

args = parser.parse_args()

arch    : str = args.arch
out_dir : str = args.out_dir or os.path.dirname(os.path.abspath(__file__))
dummy_nn_path : str = args.write_dummy_nn
original_arch : str = arch

def strip_prefix_ci(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.upper().startswith(prefix) else text

SQ_NB = 81
FILE_NB = 9
FE_END = 1548
F_KING = FE_END
E_KING = F_KING + SQ_NB
FE_END2 = E_KING + SQ_NB

FEATURE_INFO = {
    "halfkp": ("HalfKP(Friend)", 0x5D69D5B8, SQ_NB * FE_END),
    "kp": ("K+P", 0, SQ_NB * 2 + FE_END),
    "ka2": ("K+A2", 0, SQ_NB * 2 + E_KING),
    "halfkpe9": ("HalfKPE9(Friend)", 0x5D69D5B8, SQ_NB * FE_END * 3 * 3),
    "halfkpvm": ("HalfKP_vm(Friend)", 0x0B6B1D9A, 5 * FILE_NB * FE_END),
    "halfka1": ("HalfKA1(Friend)", 0x5F134CB8, SQ_NB * FE_END2),
    "halfkahm1": ("HalfKA_hm1(Friend)", 0x7F134CB8, 5 * FILE_NB * FE_END2),
    "halfka2": ("HalfKA2(Friend)", 0x5F234CB8, SQ_NB * E_KING),
    "halfkahm2": ("HalfKA_hm2(Friend)", 0x7F234CB8, 5 * FILE_NB * E_KING),
}

# makefileで指定したエディション名そのままかも知れないので削除。
arch = strip_prefix_ci(arch, "YANEURAOU_ENGINE_")
arch = strip_prefix_ci(arch, "NNUE_")

arch_upper_for_validation = arch.replace('-', '_').upper()
if "SFNNWOP" in arch_upper_for_validation:
    print("Error! : SFNNWOP architecture names are no longer supported. Use SFNN1536, SFNN_... without suffix, or SFNN_..._k3k3 / SFNN_..._king3_by_king3.")
    raise SystemExit(1)

if "LS9" in arch_upper_for_validation.split('_'):
    print("Error! : ls9 is no longer supported. Use no suffix, hand4/16/64/64z/256/1024, k3k3, k9k9, k9k9z, k13k13z, k21k21, k29k29, or their long names.")
    raise SystemExit(1)

# 出力ファイル名
filename = arch + ".h"

# 出力file path
out_path = os.path.join(out_dir, filename)

print(f"output file path  : {out_path}")

# 大文字化して、'-'を'_'に置換したアーキテクチャ名
arch   = arch.replace('-','_')
arch   = arch.upper()

print(f"architecture name : {arch}")

# if os.path.exists(out_path):
#     print("Warning : file already exists. stop.")
#     exit()
#  🤔 ファイルがすでに存在していても上書きしたほうがいいと思う。

arches = arch.split('_')
if len(arches) <= 3 :
    # アーキテクチャ名は、アンダースコアは3つ以上ないと駄目。
    print("Error! : architecture name must be like halfkp_256x2-32-32 or kp_256x2-32-32 halfkpvm_256x2_32_32")
    raise SystemExit(1)

# 📝 SFNN_halfkahm2_1536-15-32-k3k3のように指定されていれば、SFNNのheaderを生成する。
#     SFNN_ka2_3072_7_64_c1024_s256x8_k3k3 のように、cN_sMxG を置くと
#     fc_0を common N + shard M x G に分割する。
#     SFNN_halfka2_1024_7_64 のようにsuffixを省略すると、単一LayerStackになる。
#     SFNN_halfka2_1024_7_64_hand64z のように、hand64zを指定すると
#     手番側/非手番側の手駒点を8段階ずつに分けた64 bucketを用いる。
#     hand4 / hand16 / hand64 / hand256 / hand1024も同様に、手番側/非手番側の手駒状態でbucketを用いる。
#     SFNN_halfka2_1024_7_64_k9k9 / k9k9z / k13k13z / k21k21 / k29k29 のように指定すると、
#     手番側/非手番側の玉位置でbucketを分ける。
#     SFNN_halfka2_1024_7_64_hand16_k3k3 / hand64z_k9k9 / hand64z_k13k13z / hand64z_k21k21 / hand64z_k29k29 のように、
#     hand bucketと複合できる。
#     SFNN_halfka2_1024_7_64_k3k3_progress8 のようにprogress2/3/4/8/16/32とも複合できる。
#     SFNN_halfka2_1024_7_64_k3k3_wsb のように、バケット名の最後にwsb (WithSharedBucket)
#     を置くと、常に選ばれる共有バケットを1個追加してバケット数をN+1にし、選択された
#     1バケットと共有バケットの出力平均を評価値にする。
SFNN = False
layer_stack_name = ""
layer_stack_count = ""
layer_stack_hand_buckets = "1"
layer_stack_hand_bucket_type = "0"
layer_stack_king_buckets = "1"
layer_stack_king_bucket_type = "0"
layer_stack_progress_buckets = "1"
layer_stack_router_name = "NONE"
layer_stack_router_mode = "0"
layer_stack_router_n = "0"
layer_stack_shared_bucket = "0"
sfnn_group_count = "1"
sfnn_common_dims = "0"
sfnn_shard_dims = "0"
sfnn_common_shard = False

    # 📝 SFNN_halfka2_1024_7_64_k3k3_routerkpabs9 のように、末尾 (合成順に関わらず常に最後に
    #    掛かる = 最下位桁) に routerkpabs<N> を置くと、tatara `--bucket-mode ...routerkpabs<N>`
    #    で学習した net 用のビルドになる。ネットワーク構造自体は無印 (ls<N>相当) と同一で、
    #    バケット「選択方式」だけが (hand/king/progressの合成値ではなく) 学習済みの
    #    RouterKPAbs (KP-absolute特徴の重み和のargmax、tatara側 `router_kpabs.rs` 参照) になる。
    #    hand/king/progressバケットと複合可能で、その場合は
    #    「(hand/king/progressの合成バケット) * N + (routerkpabsの選択結果)」の順で合成する
    #    (routerは常に最後 = 最下位桁)。
    #
    # 📝 SFNN_halfka2_1024_7_64_k3k3_routerft8ft8 のように末尾に routerft<R>ft<R> (Rは正の整数、
    #    末尾は"ft<R>"が2回連続) を置くと、tatara `--bucket-mode ...routerft<R>ft<R>` で学習した
    #    net 用のビルドになる。この場合は routerkpabs と異なり **ネットワーク構造自体が変わる** —
    #    FeatureTransformerの片視点あたりの (pairwise-multiply前の) accumulator幅が
    #    `kInputDims + R` に広がる (router の生スコアR個がFTの同じaccumulator/重み行列に同居し、
    #    評価関数本体のFTと一緒に差分計算される; 詳細は `nnue_feature_transformer.h` の
    #    `TANUKI_ROUTER_ARCH_FTBYFT` 分岐を参照)。バケット数は R*R (STM側R通り×NSTM側R通りの
    #    argmaxの組み合わせ)。hand/king/progressバケットと複合可能で、routerkpabsと同様に
    #    常に最後 (最下位桁) に合成する。
    #
    # 📝 SFNN_halfka2_1024_7_64_k3k3_progress8_wsb のように、バケット名の最後に
    #    wsb (WithSharedBucket) を置くと、hand/king/progress/router の合成バケット数
    #    (=N) に「常に選ばれる共有バケット」を1個追加して N+1 バケットの net にする。
    #    共有バケットの重み (LayerStack配列の末尾、index N) は局面に関わらず常に
    #    forward され、通常どおり選択された1バケット (index 0..N-1) と共有バケット
    #    (index N) の出力を平均したものが評価値になる (`evaluate_nnue.cpp` の
    #    `ComputeScore` 参照)。バケット「選択方式」自体 (hand/king/progress/router) は
    #    無印と変わらない。wsb はどのトークンとも複合できるが、常に最後 (最下位の
    #    合成順ではなく、文字列上も最後) に置く必要がある。
    #
# 📝 routerkpabs と routerft<R>ft<R> は互いに排他 (同時指定不可)。router系は合計で最大1個。
# 📝 wsb は他の全トークンと複合可能 (排他なし)。ただしバケット名の最後のトークンでなければ
#    ならない (末尾以外に置くとエラー)。
def parse_sfnn_layer_stack_spec(layer_stack_spec):
    if layer_stack_spec == "":
        return "NONE", "1", "1", "0", "1", "0", "1", "NONE", "1", "0", "0"

    normalized = layer_stack_spec
    for long_name, short_name in {
            "KING3_BY_KING3": "K3K3",
            "KING9_BY_KING9": "K9K9",
            "KING9Z_BY_KING9Z": "K9K9Z",
            "KING9ZONE_BY_KING9ZONE": "K9K9Z",
            "KING13Z_BY_KING13Z": "K13K13Z",
            "KING13ZONE_BY_KING13ZONE": "K13K13Z",
            "KING21_BY_KING21": "K21K21",
            "KING29_BY_KING29": "K29K29",
        }.items():
        normalized = normalized.replace(long_name, short_name)

    hand_buckets = 1
    king_buckets = 1
    progress_buckets = 1
    hand_name = ""
    hand_type = 0
    king_name = ""
    king_type = 0
    progress_name = ""
    router_name = ""
    router_buckets = 1
    # 0 = NNUE_SFNN_ROUTER_MODE_NONE, 1 = KPABS, 2 = FTFT
    router_mode = 0
    shared_bucket = 0

    hand_map = {
        "HAND4": (4, 1),
        "HAND16": (16, 2),
        "HAND64": (64, 3),
        "HAND64Z": (64, 4),
        "HAND256": (256, 5),
        "HAND1024": (1024, 6),
    }
    king_map = {
        "K3K3": (9, 1),
        "K9K9": (81, 2),
        "K21K21": (21 * 21, 3),
        "K29K29": (29 * 29, 4),
        "K9K9Z": (81, 5),
        "K13K13Z": (13 * 13, 6),
    }
    progress_values = {2, 3, 4, 8, 16, 32}
    router_kpabs_re = re.compile(r"^ROUTERKPABS(\d+)$")
    router_ftft_re = re.compile(r"^ROUTERFT(\d+)FT(\d+)$")

    tokens = [t for t in normalized.split("_") if t]
    for pos, token in enumerate(tokens):
        m_kpabs = router_kpabs_re.match(token)
        m_ftft = router_ftft_re.match(token)
        if token == "WSB":
            if pos != len(tokens) - 1:
                print(f"Error! : wsb (WithSharedBucket) must be the last token in {layer_stack_spec}.")
                raise SystemExit(1)
            shared_bucket = 1
        elif token in hand_map:
            if hand_buckets != 1:
                print(f"Error! : duplicate SFNN hand bucket in {layer_stack_spec}.")
                raise SystemExit(1)
            hand_name = token
            hand_buckets, hand_type = hand_map[token]
        elif token in king_map:
            if king_buckets != 1:
                print(f"Error! : duplicate SFNN king bucket in {layer_stack_spec}.")
                raise SystemExit(1)
            king_name = token
            king_buckets, king_type = king_map[token]
        elif token.startswith("PROGRESS"):
            raw = token[len("PROGRESS"):]
            if not raw.isdigit() or int(raw) not in progress_values:
                print(f"Error! : progress bucket must be progress2/3/4/8/16/32 , got {token}.")
                raise SystemExit(1)
            if progress_buckets != 1:
                print(f"Error! : duplicate SFNN progress bucket in {layer_stack_spec}.")
                raise SystemExit(1)
            progress_name = token
            progress_buckets = int(raw)
        elif m_kpabs:
            if router_mode != 0:
                print(f"Error! : router bucket (routerkpabs/routerft<R>ft<R>) may appear at most once in {layer_stack_spec}.")
                raise SystemExit(1)
            n = int(m_kpabs.group(1))
            if n < 1:
                print(f"Error! : routerkpabs<N> requires N >= 1 , got {token}.")
                raise SystemExit(1)
            router_name = token
            router_mode = 1
            router_buckets = n
        elif m_ftft:
            if router_mode != 0:
                print(f"Error! : router bucket (routerkpabs/routerft<R>ft<R>) may appear at most once in {layer_stack_spec}.")
                raise SystemExit(1)
            r_a, r_b = int(m_ftft.group(1)), int(m_ftft.group(2))
            if r_a != r_b:
                print(f"Error! : routerft<R>ft<R> requires both R to match (got {token}).")
                raise SystemExit(1)
            if r_a < 1:
                print(f"Error! : routerft<R>ft<R> requires R >= 1 , got {token}.")
                raise SystemExit(1)
            router_name = token
            router_mode = 2
            router_buckets = r_a * r_a
        else:
            print(f"Error! : unknown SFNN layer stack token {token} in {layer_stack_spec}.")
            print("Error! : SFNN layer stack tokens are hand4/16/64/64z/256/1024, k3k3/k9k9/k9k9z/k13k13z/k21k21/k29k29, progress2/3/4/8/16/32, routerkpabs<N>, routerft<R>ft<R>, and wsb.")
            raise SystemExit(1)

    canonical = "_".join([name for name in [hand_name, king_name, progress_name, router_name] if name])
    if canonical == "":
        canonical = "NONE"
    if shared_bucket:
        # wsb は文字列上も常に最後に置く (上のtoken loopで既に位置を検証済み)。
        canonical = canonical + "_WSB" if canonical != "NONE" else "WSB"

    # router は常に最後 (最下位桁) に合成するので、layer_count (=kLayerStacks) は
    # hand*king*progress の積に router_buckets を最後に掛けたものになる。
    # wsb はその積に対して「常に選ばれる共有バケット」を1個追加するので、最後に+1する
    # (共有バケットの index は必ず layer_count-1 = 合成後バケット数そのもの)。
    layer_count = hand_buckets * king_buckets * progress_buckets * router_buckets + shared_bucket
    router_r = 0
    if router_mode == 2:
        # routerft<R>ft<R> の R (kRouterFtByFtR)。routerkpabs / router無しでは 0。
        router_r = int(router_ftft_re.match(router_name).group(1))
    router_n = router_buckets if router_mode == 1 else router_r
    return (canonical, str(layer_count), str(hand_buckets), str(hand_type),
        str(king_buckets), str(king_type), str(progress_buckets),
        router_name if router_name else "NONE", str(router_mode), str(router_n),
        str(shared_bucket))

def sfnn_uses_shortcut(hidden1_dims: int) -> bool:
    if hidden1_dims % 8 == 7:
        return True
    if hidden1_dims % 8 == 0:
        return False
    print(f"Error : SFNN H1 must be 8n with no shortcut, or 8n-1 with shortcut. H1={hidden1_dims}.")
    raise SystemExit(1)

def sfnn_hidden1_output_dims(hidden1_dims: int) -> int:
    return hidden1_dims + (1 if sfnn_uses_shortcut(hidden1_dims) else 0)

if arches[0].startswith("SFNN"):
    SFNN = True
    if len(arches) < 5:
        print("Error! : SFNN architecture name must be like SFNN_halfka2_1024_7_64, SFNN_halfkahm2_1536-15-32-k3k3, or SFNN_ka2_3072_7_64_c1024_s256x8_k3k3")
        raise SystemExit(1)

    layer_stack_start = 5
    if len(arches) > 5 and arches[5].startswith("C"):
        common_raw = arches[5][1:]
        if not common_raw.isdigit():
            print(f"Error! : SFNN common token must be like c0 or c1024 , got {arches[5]}.")
            raise SystemExit(1)
        if len(arches) <= 6 or not arches[6].startswith("S"):
            print("Error! : SFNN common+shard architecture requires shard token like s256x8.")
            raise SystemExit(1)
        shard_spec = arches[6][1:]
        shard_parts = shard_spec.split("X")
        if (len(shard_parts) != 2 or not shard_parts[0].isdigit()
                or not shard_parts[1].isdigit() or int(shard_parts[0]) <= 0
                or int(shard_parts[1]) <= 1):
            print(f"Error! : SFNN shard token must be like s256x8 , got {arches[6]}.")
            raise SystemExit(1)
        sfnn_common_dims = common_raw
        sfnn_shard_dims = shard_parts[0]
        sfnn_group_count = shard_parts[1]
        sfnn_common_shard = True
        layer_stack_start = 7
    layer_stack_spec = "_".join(arches[layer_stack_start:]) if len(arches) > layer_stack_start else ""
    (layer_stack_name, layer_stack_count, layer_stack_hand_buckets,
        layer_stack_hand_bucket_type,
        layer_stack_king_buckets, layer_stack_king_bucket_type,
        layer_stack_progress_buckets,
        layer_stack_router_name, layer_stack_router_mode,
        layer_stack_router_n,
        layer_stack_shared_bucket) = parse_sfnn_layer_stack_spec(layer_stack_spec)

    arches = [arches[1], arches[2], arches[3], arches[4], layer_stack_count]

# ============================================================
#                        includes
# ============================================================

if SFNN:
    header = f"""
    // SFNN without PSQT architecture

    #ifndef CLASSIC_NNUE_SFNN_{arch}_H_INCLUDED
    #define CLASSIC_NNUE_SFNN_{arch}_H_INCLUDED
    """
else:
    header = f"""
    // Definition of input features and network structure used in NNUE evaluation function
    // NNUE評価関数で用いる入力特徴量とネットワーク構造の定義
    #ifndef NNUE_{arch}_H_INCLUDED
    #define NNUE_{arch}_H_INCLUDED
    """

# ============================================================
#                     input features
# ============================================================

# アーキテクチャ名のアンダースコアでsplitした1つ目は入力特徴量。
# 現在サポートしている入力特徴量は、
#   halfkp
#   kp
#   ka2
#   halfkpe9
#   halfkpvm
#   halfka1
#   halfkahm1
#   halfka2
#   halfkahm2

input_feature = arches[0].lower()

print(f"input feature     : {input_feature}")

raw_feature_name, raw_feature_hash, raw_feature_dims = FEATURE_INFO.get(input_feature, ("", 0, 0))

header += f"""
    #include "../features/feature_set.h"
    """

if input_feature == "halfkp":

    header += f"""
    #include "../features/half_kp.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKP<Features::Side::kFriend>>;
    """

elif input_feature == "kp":

    header += f"""
    #include "../features/k.h"
    #include "../features/p.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<Features::K, Features::P>;
    """

elif input_feature == "ka2":

    header += f"""
    #include "../features/k.h"
    #include "../features/a2.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<Features::K, Features::A2>;
    """

elif input_feature == "halfkpe9":

    header += f"""
    #include "../features/half_kpe9.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKPE9<Features::Side::kFriend>>;
    """

elif input_feature == "halfkpvm":

    header += f"""
    #include "../features/half_kp_vm.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKP_vm<Features::Side::kFriend>>;
    """

elif input_feature == "halfka1":

    header += f"""
    #include "../features/half_ka1.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKA1<Features::Side::kFriend>>;
    """

elif input_feature == "halfkahm1":

    header += f"""
    #include "../features/half_ka_hm1.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKA_hm1<Features::Side::kFriend>>;
    """

elif input_feature == "halfka2":

    header += f"""
    #include "../features/half_ka2.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKA2<Features::Side::kFriend>>;
    """

elif input_feature == "halfkahm2":

    header += f"""
    #include "../features/half_ka_hm2.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKA_hm2<Features::Side::kFriend>>;
    """

else:
    # 知らない入力特徴量だった。
    print(f"Error : input feature {input_feature} is not supported.")
    raise SystemExit(1)

if SFNN:
    header += """
    #include "sfnn_network.h"

    namespace YaneuraOu {
    namespace Eval::NNUE {

    // Input features used in evaluation function
    // 評価関数で用いる入力特徴量
    """

else:    

    header += """
    #include "../layers/input_slice.h"
    #include "../layers/affine_transform.h"
    #include "../layers/affine_transform_sparse_input.h"
    #include "../layers/clipped_relu.h"

    namespace YaneuraOu {
    namespace Eval::NNUE {

    // Input features used in evaluation function
    // 評価関数で用いる入力特徴量
    """

header += raw_features

# ============================================================
#                     hidden layers
# ============================================================

# レイヤ情報
# 例えば、"256x2_32_32" ならば ["256x2","32","32"]のように分解される。
#   (SFNNで) "1536-15-32-k3k3" なら ["1536","15","32","9"]のように分解される。(はず)
layers = arches[1:]
layers[0] = layers[0].lower()

if SFNN:
    if len(layers) != 4:
        print(f"Error : layers must be like 1536-15-32-k3k3 , layers = {layers}.")
        raise SystemExit(1)

    hidden1_dims = int(layers[1])
    hidden1_output_dims = sfnn_hidden1_output_dims(hidden1_dims)
    hidden1_uses_shortcut = sfnn_uses_shortcut(hidden1_dims)

    if not sfnn_group_count.isdigit():
        print(f"Error : SFNN group count must be an integer , group = {sfnn_group_count}.")
        raise SystemExit(1)

    if sfnn_common_shard:
        transformed_dims = int(layers[0])
        common_dims = int(sfnn_common_dims)
        shard_dims = int(sfnn_shard_dims)
        group_count = int(sfnn_group_count)
        if common_dims + shard_dims * group_count != transformed_dims:
            print(f"Error : common+shard SFNN requires common + shard * group == transformed dimensions. common={common_dims}, shard={shard_dims}, group={group_count}, dims={transformed_dims}.")
            raise SystemExit(1)
        if hidden1_output_dims % group_count != 0:
            print(f"Error : common+shard SFNN requires fc0 output dimensions divisible by group count. fc0_output={hidden1_output_dims}, group={group_count}.")
            raise SystemExit(1)
        if common_dims % 64 != 0:
            print(f"Error : common+shard SFNN requires common dimensions to be a multiple of 64. common={common_dims}.")
            raise SystemExit(1)
        if shard_dims % 64 != 0:
            print(f"Error : common+shard SFNN requires shard dimensions to be a multiple of 64. shard={shard_dims}.")
            raise SystemExit(1)

    print(f"layers feature    : {layers}")

    small_sfnn_ft_macro = ""
    if int(layers[0]) < 128:
        small_sfnn_ft_macro = "#define NNUE_SMALL_SFNN_FT"

    # routerft<R>ft<R> のときだけ、FeatureTransformer側 (kTransformedFeatureDimensions =
    # per-perspective, pre-pairwise-multiply の accumulator幅) が fc_0 の入力幅 (kInputDims)
    # より kRouterFtByFtR (=R) だけ広くなる — router の生スコア R個 (perspectiveごと) が
    # 同じ accumulator/重み行列に同居するため (詳細は nnue_feature_transformer.h の
    # TANUKI_ROUTER_ARCH_FTBYFT 分岐を参照)。routerkpabs / router無しでは従来通り一致する。
    router_is_ftft = layer_stack_router_mode == "2"
    router_ft_r = int(layer_stack_router_n) if router_is_ftft else 0
    main_dims = int(layers[0])
    transformed_dims_value = main_dims + router_ft_r

    router_macro_block = f"""
        #define NNUE_SFNN_ROUTER_MODE_NONE 0
        #define NNUE_SFNN_ROUTER_MODE_KPABS 1
        #define NNUE_SFNN_ROUTER_MODE_FTFT 2
        #define NNUE_SFNN_ROUTER_MODE {layer_stack_router_mode}
        // routerkpabsではバケット数そのもの、routerft<R>ft<R>ではRを表す (バケット数はR*R)。
        // 無印/router無しでは 0。
        #define NNUE_SFNN_ROUTER_N {layer_stack_router_n}
    """
    if router_is_ftft:
        router_macro_block += f"""
        // `--bucket-mode ...routerft{{R}}ft{{R}}` (tatara) 用の net。FeatureTransformer の
        // accumulator に同居する router の生スコア数 (perspectiveごとに R 個)。
        // routerkpabs / router無しビルドでは定義しない (0扱い)。
        #define TANUKI_ROUTER_ARCH_FTBYFT
        constexpr IndexType kRouterFtByFtR = {router_ft_r};
    """

    header += f"""
        // Number of input feature dimensions after conversion
        // 変換後の入力特徴量の次元数
        constexpr IndexType kTransformedFeatureDimensions = {transformed_dims_value};

        // 小幅SFNN専用。従来幅のSFNNでは定義せず、既存の高速経路をそのまま使う。
        {small_sfnn_ft_macro}

        // Number of networks stored in the evaluation file
        constexpr int LayerStacks = {layers[3]};

        #define NNUE_SFNN_HAND_BUCKETS {layer_stack_hand_buckets}
        #define NNUE_SFNN_HAND_BUCKET_TYPE {layer_stack_hand_bucket_type}
        #define NNUE_SFNN_KING_BUCKETS {layer_stack_king_buckets}
        #define NNUE_SFNN_KING_BUCKET_TYPE {layer_stack_king_bucket_type}
        #define NNUE_SFNN_PROGRESS_BUCKETS {layer_stack_progress_buckets}
        {router_macro_block}

        // wsb (WithSharedBucket)。1のとき、LayerStacks (={layers[3]}) の末尾index
        // (=LayerStacks-1) は hand/king/progress/routerの選択に関わらず常に評価される
        // 共有バケットで、選択された1バケットとの出力平均を評価値にする
        // (`evaluate_nnue.cpp` の `ComputeScore` 参照)。0なら従来通り選択された1バケット
        // のみを使う。
        #define NNUE_SFNN_USE_SHARED_BUCKET {layer_stack_shared_bucket}

        // Number of groups for the first affine layer of SFNN.
        // common+shard fc_0でのみ2以上になる。
        constexpr IndexType kHidden1GroupCount = {sfnn_group_count};

        // common+shard fc_0 settings. kHidden1ShardDimensions is per shard.
        constexpr bool kHidden1UsesCommonShard = {"true" if sfnn_common_shard else "false"};
        constexpr IndexType kHidden1CommonDimensions = {sfnn_common_dims};
        constexpr IndexType kHidden1ShardDimensions = {sfnn_shard_dims};

        // 各層の次元数
        // routerft<R>ft<R> のときは kTransformedFeatureDimensions (FT accumulator幅) より
        // R だけ狭い (router の生スコア分を除いた実際の特徴量次元)。それ以外は一致する。
        constexpr IndexType kInputDims   = {main_dims};
        constexpr IndexType kHidden1Dims = {layers[1]};
        constexpr bool kUseShortcut = {"true" if hidden1_uses_shortcut else "false"};
        constexpr IndexType kHidden1OutputDims = kHidden1Dims + (kUseShortcut ? 1 : 0);
        constexpr IndexType kHidden2Dims = {layers[2]};                              
    """

else:

    if len(layers) != 3 or len(layers[0].split('x')) != 2:
        print(f"Error : layers must be like 256x2-32-32 , layers = {layers}.")
        raise SystemExit(1)

    first_layer = layers[0].split('x')

    print(f"layers feature    : {layers}")

    header += f"""
        // Number of input feature dimensions after conversion
        // 変換後の入力特徴量の次元数
        constexpr IndexType kTransformedFeatureDimensions = {first_layer[0]};

        namespace Layers {{

            // Define network structure
            // ネットワーク構造の定義
            using InputLayer = InputSlice<kTransformedFeatureDimensions * {first_layer[1]}>;
            using HiddenLayer1 = ClippedReLU<AffineTransformSparseInput<InputLayer, {layers[1]}>>;
            using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, {layers[2]}>>;
            using OutputLayer = AffineTransform<HiddenLayer2, 1>;

        }}  // namespace Layers
    """

# ============================================================
#                     output layer
# ============================================================

if SFNN:
    fc_0_type = "Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1OutputDims>"
    group_count = int(sfnn_group_count)
    if sfnn_common_shard:
        fc_0_type = "Layers::AffineTransformCommonShardInputExplicit<kInputDims, kHidden1OutputDims, kHidden1CommonDimensions, kHidden1ShardDimensions, kHidden1GroupCount>"
    group_input_dims = int(sfnn_shard_dims) if sfnn_common_shard else 0
    enable_common_shard_sfnn_accumulator_propagate = (
        sfnn_common_shard and group_count % 2 == 0 and group_input_dims % 64 == 0
    )
    enable_sparse_sfnn_accumulator_propagate = (
        False
        and not sfnn_common_shard
        and hidden1_output_dims == 8
        and int(layers[0]) % 128 == 0
    )
    sfnn_accumulator_propagate_macro = ""
    if enable_common_shard_sfnn_accumulator_propagate:
        sfnn_accumulator_propagate_macro = "#define NNUE_HAS_SFNN_ACCUMULATOR_PROPAGATE"
    elif enable_sparse_sfnn_accumulator_propagate:
        sfnn_accumulator_propagate_macro = "#define NNUE_HAS_SFNN_ACCUMULATOR_PROPAGATE"

    structure_string = (
        "SFNN-1536"
        if input_feature == "halfkahm2"
        and layers == ["1536", "15", "32", "9"]
        and layer_stack_name == "K3K3"
        else arch
    )

    header += f"""
        {sfnn_accumulator_propagate_macro}

        using Fc0Layer = {fc_0_type};
        using NetworkBase = SfnnNetwork<Fc0Layer, kInputDims, kHidden1Dims, kHidden2Dims, kUseShortcut>;

        struct Network : NetworkBase {{
            static std::string GetStructureString() {{
                return "{structure_string}";
            }}
        }};

    }}  // namespace Eval::NNUE
    }}  // namespace YaneuraOu

    #endif // CLASSIC_NNUE_{arch}_H_INCLUDED
    """

    # 💡 GetStructureString()で異なる文字列を返すと別のアーキテクチャとみなされてしまう。

else:
    header += f"""
        using Network = Layers::OutputLayer;

    }} // namespace Eval::NNUE
    }} // namespace YaneuraOu

    #endif // #ifndef NNUE_{arch}_H_INCLUDED
    """

if out_dir:
    os.makedirs(out_dir, exist_ok=True)

with open(out_path, "w", encoding = 'utf-8') as f:
    f.write(dedent4(header))

print("..done!")

if dummy_nn_path:
    dummy_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnue_dummy_gen.py")
    subprocess.run([
        sys.executable,
        dummy_script,
        original_arch,
        dummy_nn_path,
        "--dummy-mode",
        args.dummy_mode,
        "--dummy-seed",
        str(args.dummy_seed),
    ], check=True)
