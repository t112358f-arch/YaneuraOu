# NNUE architecture header generator
#
#  NNUE評価関数のarchitecture headerを動的に生成するPythonで書かれたスクリプト。
# 

import argparse
import os

def dedent4(text: str) -> str:
    # 各行の先頭4文字（スペース4つ）を削除して結合し直す
    # 行が4文字未満、あるいはスペースでない場合を考慮して lstrip でも可
    return "\n".join(line[4:] if line.startswith("    ") else line 
                        for line in text.strip("\n").splitlines())


print("NNUE architecture header generator by yaneurao V1.02 , 2026/01/31")

parser = argparse.ArgumentParser(description="NNUEのarchitecture headerを生成する。")
parser.add_argument('arch', type=str, nargs='?', default="halfkp_256x2-32-32", help="architectureを指定する。例) halfkp_1024x2-8-64, YANEURAOU_ENGINE_NNUE_HALFKP_1024X2_16_32とか")
parser.add_argument('out_dir', type=str, nargs='?', default="", help="出力先のフォルダを指定する。例) /source/eval/nnue/architectures/")
parser.add_argument('--l1', type=str, default="", help="SFNNwoP_V3専用。bucketごとのL1(kHidden1)サイズをカンマ区切り9個の自然数で指定する。例) 15,15,15,20,20,20,25,25,25")
parser.add_argument('--l2', type=str, default="", help="SFNNwoP_V3専用。bucketごとのL2(kHidden2)サイズをカンマ区切り9個の自然数で指定する。例) 32,32,32,40,40,40,48,48,48")

args = parser.parse_args()

arch    : str = args.arch
out_dir : str = args.out_dir

def strip_prefix_ci(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.upper().startswith(prefix) else text

# makefileで指定したエディション名そのままかも知れないので削除。
arch = strip_prefix_ci(arch, "YANEURAOU_ENGINE_")
arch = strip_prefix_ci(arch, "NNUE_")

arch_upper_for_validation = arch.replace('-', '_').upper()

# ============================================================
#   SFNNwoP_V3: bucketごとにl1/l2のサイズを個別指定できるlayerstack
# ============================================================
#
# 例) SFNNwoP_V3-1536 , SFNNwoP_V3 (ft省略時は1536)
# LS_BUCKET_MODE (kingrank9 / progress8kpabs / progress9kpabs) はUSIオプションで
# 実行時に切り替えるため、他のSFNNアーキテクチャのような "_k3k3" 等のbucket方式
# suffixはarchitecture名に付けない。bucket数は常に9固定。
NUM_BUCKETS_V3 = 9

def _parse_bucket_list(csv: str, default: int, opt_name: str) -> list:
    csv = csv.strip()
    if csv == "":
        return [default] * NUM_BUCKETS_V3
    parts = [p.strip() for p in csv.split(',')]
    if len(parts) != NUM_BUCKETS_V3:
        print(f"Error! : {opt_name} must be {NUM_BUCKETS_V3} comma-separated natural numbers, got {len(parts)}: '{csv}'")
        raise SystemExit(1)
    values = []
    for p in parts:
        if not p.isdigit() or int(p) <= 0:
            print(f"Error! : {opt_name} entries must be natural numbers (>0), got '{p}' in '{csv}'")
            raise SystemExit(1)
        values.append(int(p))
    return values

if arch_upper_for_validation.startswith("SFNNWOP_V3"):
    rest = arch_upper_for_validation[len("SFNNWOP_V3"):].lstrip('_').lstrip('-')
    ft_out_v3 = int(rest) if rest != "" else 1536
    if ft_out_v3 <= 0 or ft_out_v3 % 128 != 0:
        print(f"Error! : SFNNwoP_V3 ft (transformed feature dimensions) must be a positive multiple of 128, got {ft_out_v3}")
        raise SystemExit(1)

    l1_list = _parse_bucket_list(args.l1, 15, "--l1")
    l2_list = _parse_bucket_list(args.l2, 32, "--l2")

    filename = arch + ".h"
    out_path = os.path.join(out_dir, filename)
    print(f"output file path  : {out_path}")
    print(f"architecture name : SFNNwoP_V3 (ft={ft_out_v3})")
    print(f"per-bucket L1     : {l1_list}")
    print(f"per-bucket L2     : {l2_list}")

    guard = f"CLASSIC_NNUE_SFNNWOP_V3_{ft_out_v3}_H_INCLUDED"

    def _bucket_hash(idx: int, l1: int, l2: int) -> int:
        # bucketごとに一意な32bit hash (ファイル内容の妥当性チェックにのみ使用、
        # 読み込み側は不一致でもwarningのみで継続する)。
        h = 0x53464E33  # 'SFN3'
        h = (h * 0x01000193 + ft_out_v3) & 0xFFFFFFFF
        h = (h * 0x01000193 + l1) & 0xFFFFFFFF
        h = (h * 0x01000193 + l2) & 0xFFFFFFFF
        h = (h * 0x01000193 + idx) & 0xFFFFFFFF
        return h

    bucket_hashes = [_bucket_hash(i, l1_list[i], l2_list[i]) for i in range(NUM_BUCKETS_V3)]
    combined_hash = 0x6333718A
    for h in bucket_hashes:
        combined_hash ^= h
    combined_hash &= 0xFFFFFFFF

    bucket_typedefs = "\n".join(
        f"using NetworkBucket{i} = NetworkBucket<{l1_list[i]}, {l2_list[i]}, {bucket_hashes[i]}u>;"
        for i in range(NUM_BUCKETS_V3)
    )
    bucket_members = "\n\t".join(f"NetworkBucket{i} b{i};" for i in range(NUM_BUCKETS_V3))
    read_calls = "bool ok = " + "\n\t\t\t&& ".join(f"b{i}.ReadParameters(stream).is_ok()" for i in range(NUM_BUCKETS_V3)) + ";"
    write_calls = "return " + "\n\t\t\t&& ".join(f"b{i}.WriteParameters(stream)" for i in range(NUM_BUCKETS_V3)) + ";"
    buffer_size_expr = "std::max({" + ", ".join(f"NetworkBucket{i}::kBufferSize" for i in range(NUM_BUCKETS_V3)) + "})"
    propagate_cases = "\n\t\t".join(
        f"case {i}: return b{i}.Propagate(transformedFeatures, buffer);" for i in range(NUM_BUCKETS_V3)
    )
    l1_list_str = ",".join(str(v) for v in l1_list)
    l2_list_str = ",".join(str(v) for v in l2_list)

    header_v3 = f"""
    // SFNNwoP_V3 : bucketごとに可変サイズのl1/l2を持つlayerstack architecture
    // (nnue_arch_gen.pyにより自動生成)
    //
    // - ft_out はbucket間で共通 ({ft_out_v3})
    // - l1/l2 はbucketごとに個別サイズ (--l1 / --l2 で指定)
    // - bucket選択 (kingrank9 / progress8kpabs / progress9kpabs) はUSIオプション
    //   LS_BUCKET_MODE で実行時に切り替える (architecture名には含めない)

    #ifndef {guard}
    #define {guard}

    #include "../features/feature_set.h"
    #include "../features/half_ka_hm2.h"

    #include <cstring>
    #include <algorithm>
    #include <string>

    #include "../layers/affine_transform_explicit.h"
    #include "../layers/affine_transform_sparse_input_explicit.h"
    #include "../layers/clipped_relu_explicit.h"
    #include "../layers/sqr_clipped_relu.h"

    namespace YaneuraOu {{
    namespace Eval::NNUE {{

    using RawFeatures = Features::FeatureSet<
        Features::HalfKA_hm2<Features::Side::kFriend>>;

    // 変換後の入力特徴量の次元数 (bucket間で共通)
    constexpr IndexType kTransformedFeatureDimensions = {ft_out_v3};

    // NnueNetworks::network[] の要素数。SFNNwoP_V3は9bucket分をNetwork 1個に
    // 集約するので常に1。実際のbucket数はkNumBuckets。
    constexpr int LayerStacks = 1;
    constexpr int kNumBuckets = {NUM_BUCKETS_V3};

    constexpr IndexType kInputDims = kTransformedFeatureDimensions;

    // bucketごとのL1/L2出力次元 (参考情報として公開)
    constexpr IndexType kHidden1DimsPerBucket[kNumBuckets] = {{ {l1_list_str} }};
    constexpr IndexType kHidden2DimsPerBucket[kNumBuckets] = {{ {l2_list_str} }};

    // 1bucket分のネットワーク。L1/L2サイズをtemplate引数化することでbucketごとに
    // 異なるサイズを持たせられる。
    template <IndexType kHidden1, IndexType kHidden2, std::uint32_t kHash>
    struct NetworkBucket {{

        Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1 + 1> fc_0;
        Layers::ClippedReLUExplicit<kHidden1 + 1> ac_0;
        Layers::SqrClippedReLU<kHidden1 + 1> ac_sqr_0;

        Layers::AffineTransformExplicit<kHidden1 * 2, kHidden2> fc_1;
        Layers::ClippedReLUExplicit<kHidden2> ac_1;

        Layers::AffineTransformExplicit<kHidden2, 1> fc_2;

        using OutputType = std::int32_t;
        static constexpr IndexType kOutputDimensions = 1;

        static constexpr std::uint32_t GetHashValue() {{ return kHash; }}

        Tools::Result ReadParameters(std::istream& stream) {{
            bool result = fc_0.ReadParameters(stream).is_ok()
                && ac_0.ReadParameters(stream).is_ok()
                && ac_sqr_0.ReadParameters(stream).is_ok()
                && fc_1.ReadParameters(stream).is_ok()
                && ac_1.ReadParameters(stream).is_ok()
                && fc_2.ReadParameters(stream).is_ok();
            return result ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
        }}

        bool WriteParameters(std::ostream& stream) const {{
            return fc_0.WriteParameters(stream)
                && ac_0.WriteParameters(stream)
                && ac_sqr_0.WriteParameters(stream)
                && fc_1.WriteParameters(stream)
                && ac_1.WriteParameters(stream)
                && fc_2.WriteParameters(stream);
        }}

        struct alignas(kCacheLineSize) Buffer {{
            alignas(kCacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
            alignas(kCacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
            alignas(kCacheLineSize) typename decltype(ac_sqr_0)::OutputType ac_sqr_0_out[CeilToMultiple<IndexType>(kHidden1 * 2, 32)];
            alignas(kCacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
            alignas(kCacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
            alignas(kCacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;
        }};

        static constexpr std::size_t kBufferSize = sizeof(Buffer);

        const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, char* buffer) const {{
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
        }}
    }};

    {bucket_typedefs}

    // 9bucket分の集約。NnueNetworksからは常にnetwork[0]の1個として扱われ、
    // 実際のbucket選択はPropagate()の引数(0..kNumBuckets-1)で行う。
    struct Network {{

        {bucket_members}

        using OutputType = std::int32_t;
        static constexpr IndexType kOutputDimensions = 1;

        static constexpr std::uint32_t GetHashValue() {{
            return {combined_hash}u;
        }}

        static std::string GetStructureString() {{
            return "SFNNwoP-V3-{ft_out_v3}-L1[{l1_list_str}]-L2[{l2_list_str}]";
        }}

        Tools::Result ReadParameters(std::istream& stream) {{
            {read_calls}
            return ok ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
        }}

        bool WriteParameters(std::ostream& stream) const {{
            {write_calls}
        }}

        static constexpr std::size_t kBufferSize = {buffer_size_expr};

        const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, char* buffer, int bucket) const {{
            switch (bucket) {{
            {propagate_cases}
            default:
                return b0.Propagate(transformedFeatures, buffer);
            }}
        }}
    }};

    }}  // namespace Eval::NNUE
    }}  // namespace YaneuraOu

    #endif // {guard}
    """

    with open(out_path, "w", encoding='utf-8') as f:
        f.write(dedent4(header_v3))

    print("..done! (SFNNwoP_V3)")
    raise SystemExit(0)

if "SFNNWOP" in arch_upper_for_validation:
    print("Error! : SFNNWOP architecture names are no longer supported. Use SFNN1536 or SFNN_..._k3k3 / SFNN_..._king3_by_king3 / SFNNwoP_V3-<ft>.")
    raise SystemExit(1)

if "LS9" in arch_upper_for_validation.split('_'):
    print("Error! : ls9 is no longer supported. Use k3k3 or king3_by_king3.")
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
SFNN = False
layer_stack_name = ""
if arches[0].startswith("SFNN"):
    SFNN = True
    if len(arches) < 6:
        print("Error! : SFNN architecture name must be like SFNN_halfkahm2_1536-15-32-k3k3")
        raise SystemExit(1)

    layer_stack_spec = "_".join(arches[5:])
    if layer_stack_spec == "K3K3" or layer_stack_spec == "KING3_BY_KING3":
        layer_stack_name = "K3K3"
        layer_stack_count = "9"
    else:
        print("Error! : SFNN layer stack must be k3k3 or king3_by_king3")
        raise SystemExit(1)

    arches = [arches[1], arches[2], arches[3], arches[4], layer_stack_count]

# ============================================================
#                        includes
# ============================================================

if SFNN:
    header = f"""
    // SFNN without PSQT 1536 architecture

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
    #include <cstring>

    #include "../layers/affine_transform_explicit.h"
    #include "../layers/affine_transform_sparse_input_explicit.h"
    #include "../layers/clipped_relu_explicit.h"
    #include "../layers/sqr_clipped_relu.h"

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

    print(f"layers feature    : {layers}")

    header += f"""
        // Number of input feature dimensions after conversion
        // 変換後の入力特徴量の次元数
        constexpr IndexType kTransformedFeatureDimensions = {layers[0]};

        // Number of networks stored in the evaluation file
        constexpr int LayerStacks = {layers[3]};

        // 各層の次元数
        constexpr IndexType kInputDims   = kTransformedFeatureDimensions;
        constexpr IndexType kHidden1Dims = {layers[1]};
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
    # `sfnn-1536.h`からそのままコピペ。
    header += f"""
        struct Network {{

            // Define network structure
            // ネットワーク構造の定義
            Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1Dims + 1> fc_0;
            Layers::ClippedReLUExplicit<kHidden1Dims + 1> ac_0;
            Layers::SqrClippedReLU<kHidden1Dims + 1> ac_sqr_0;

            Layers::AffineTransformExplicit<kHidden1Dims * 2, kHidden2Dims> fc_1;
            Layers::ClippedReLUExplicit<kHidden2Dims> ac_1;
            
        Layers::AffineTransformExplicit<kHidden2Dims, 1> fc_2;

            using OutputType = std::int32_t;
            static constexpr IndexType kOutputDimensions = 1;

            // Hash値などは適宜実装
            static constexpr std::uint32_t GetHashValue() {{
                return 0x6333718Au;
            }}

            static std::string GetStructureString() {{
                return "{'SFNN-1536' if input_feature == 'halfkahm2' and layers == ['1536', '15', '32', '9'] and layer_stack_name == 'K3K3' else arch}";
            }}

            Tools::Result ReadParameters(std::istream& stream) {{
                bool result = fc_0.ReadParameters(stream).is_ok()
                    && ac_0.ReadParameters(stream).is_ok()
                    && ac_sqr_0.ReadParameters(stream).is_ok()
                    && fc_1.ReadParameters(stream).is_ok()
                    && ac_1.ReadParameters(stream).is_ok()
                    && fc_2.ReadParameters(stream).is_ok();
                return result ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
            }}

            bool WriteParameters(std::ostream& stream) const {{
                return fc_0.WriteParameters(stream)
                    && ac_0.WriteParameters(stream)
                    && ac_sqr_0.WriteParameters(stream)
                    && fc_1.WriteParameters(stream)
                    && ac_1.WriteParameters(stream)
                    && fc_2.WriteParameters(stream);
            }}

            struct alignas(kCacheLineSize) Buffer {{
                alignas(kCacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
                alignas(kCacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
                alignas(kCacheLineSize) typename decltype(ac_sqr_0)::OutputType ac_sqr_0_out[CeilToMultiple<IndexType>(kHidden1Dims * 2, 32)];
                alignas(kCacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
                alignas(kCacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
                alignas(kCacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;
            }};

            static constexpr std::size_t kBufferSize = sizeof(Buffer);

            const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, char* buffer) const {{
                auto& buf = *reinterpret_cast<Buffer*>(buffer);
                std::memset(buf.ac_sqr_0_out, 0, sizeof(buf.ac_sqr_0_out));

                fc_0.Propagate(transformedFeatures, buf.fc_0_out);
                ac_0.Propagate(buf.fc_0_out, buf.ac_0_out);
                ac_sqr_0.Propagate(buf.fc_0_out, buf.ac_sqr_0_out);
                std::memcpy(buf.ac_sqr_0_out + kHidden1Dims, buf.ac_0_out,
                    kHidden1Dims * sizeof(typename decltype(ac_0)::OutputType));
                fc_1.Propagate(buf.ac_sqr_0_out, buf.fc_1_out);
                ac_1.Propagate(buf.fc_1_out, buf.ac_1_out);
                fc_2.Propagate(buf.ac_1_out, buf.fc_2_out);

                // add shortcut term
                buf.fc_2_out[0] += buf.fc_0_out[kHidden1Dims];

                return buf.fc_2_out;
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

with open(out_path, "w", encoding = 'utf-8') as f:
    f.write(dedent4(header))

print("..done!")
