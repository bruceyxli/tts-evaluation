# TTS Evaluation Project Notes

## 项目架构

本项目使用**独立 conda 环境**运行不同的 TTS 模型，避免依赖冲突。

### 环境映射

| 模型类型 | Conda 环境 | 说明 |
|---------|-----------|------|
| `glm_tts` | `tts-glm` | GLM-TTS 基础版 |
| `glm_tts_rl` | `tts-glm` | GLM-TTS + RL 权重 |
| `qwen_tts` | `tts-qwen` | Qwen3-TTS (CustomVoice) |
| `qwen_tts_vc` | `tts-qwen` | Qwen3-TTS Voice Cloning (Base 模型) |
| `qwen_tts_vllm` | `tts-qwen-vllm` | **vLLM 加速** (Linux only) |
| `qwen_tts_vllm_vc` | `tts-qwen-vllm` | vLLM Voice Cloning (Linux only) |
| `cosyvoice` | `tts-cosyvoice` | CosyVoice (Fun-CosyVoice3-0.5B) |
| `cosyvoice_rl` | `tts-cosyvoice` | CosyVoice + RL 权重 (llm.rl.pt) |
| `cosyvoice_vllm` | `tts-cosyvoice-vllm` | vLLM 加速 (Linux only) |

### 关键文件

- `main.py` - 主入口，通过 subprocess 自动切换环境
- `model_runner.py` - 在隔离环境中执行的模型加载和合成脚本
- `model_runner_vllm.py` - vLLM-Omni 专用推理脚本 (高性能)
- `setup_models.py` - 创建 conda 环境和下载模型
- `envs/*.yaml` - 各环境的 conda 配置文件

## 运行流程

```
main.py
   │
   ├── 读取 config.yaml
   ├── 遍历每个模型
   │      │
   │      ├── 创建临时 JSON 文件（config, tasks）
   │      ├── conda run -n <env> python model_runner.py ...
   │      ├── model_runner.py 在对应环境中：
   │      │      ├── 加载模型
   │      │      ├── 合成所有文本
   │      │      └── 输出 JSON 结果到 stdout
   │      └── main.py 收集结果
   │
   └── 保存 metrics.json / detailed_results.json
```

## 注意事项

### CosyVoice 路径配置

CosyVoice 需要额外的 sys.path 配置：
```python
sys.path.insert(0, str(repo_path))
sys.path.insert(0, str(repo_path / "third_party" / "Matcha-TTS"))
```

### CosyVoice RL 权重

RL 版本会加载 `llm.rl.pt` 文件：
```python
rl_weights = model_path / "llm.rl.pt"
model.model.llm.load_state_dict(state_dict, strict=False)
```

### 模型默认路径

- GLM-TTS: `./GLM-TTS/ckpt`
- Qwen-TTS: 从 HuggingFace 自动下载
- CosyVoice: `./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B`

### 添加新模型

1. 在 `envs/` 创建环境 YAML
2. 在 `setup_models.py` 的 `ENVS` 字典添加配置
3. 在 `model_runner.py` 添加 `load_xxx()` 和 `synthesize_xxx()` 函数
4. 在 `main.py` 的 `MODEL_ENV_MAP` 添加映射

## 常用命令

```bash
# 查看环境状态
python setup_models.py --list

# 安装环境
python setup_models.py --all
python setup_models.py --glm --qwen

# 运行测试
python main.py                        # 全部模型
python main.py -m glm_tts qwen_tts    # 指定模型
python main.py -s scripts.json        # 自定义文本

# 手动在某环境运行
conda run -n tts-glm python -c "import torch; print(torch.cuda.is_available())"
```

## 评估指标

### RTF (Real-Time Factor)
- 计算公式: `generation_time / audio_duration`
- RTF < 1.0 表示合成速度快于实时播放
- RTF = 1.0 表示实时
- RTF > 1.0 表示慢于实时

### Stream First Token Latency
- 流式合成时，首个音频 chunk 返回的延迟时间
- 反映模型的响应速度，对实时交互场景很重要
- 对于不支持流式的模型返回 `null`
- CosyVoice 原生支持流式

### Streaming 支持现状

| 模型 | 本地 Streaming | 说明 |
|------|---------------|------|
| CosyVoice | ✅ 支持 | `inference_bistream()` 原生支持 |
| GLM-TTS | ⚠️ 可实现 | 有 `token2wav_stream()` 但需要额外集成 |
| Qwen3-TTS | ❌ 不支持 | qwen-tts 库不暴露 streaming API |

#### Qwen3-TTS Streaming 说明

虽然 Qwen3-TTS 模型架构支持 "Dual-Track Hybrid Streaming" (双轨混合流式)，但 **qwen-tts Python 库不支持本地 streaming 输出**：

- `generate_custom_voice()` 和 `generate_voice_clone()` 返回完整音频，不是 generator
- `non_streaming_mode` 参数仅控制**文本输入**的模拟流式，不影响音频输出
- Qwen 官方确认：qwen-tts 库专注于 "quick demos, rapid prototyping, and batch offline inference"

**获取 Streaming 的方式**：
1. **DashScope Real-time API** (阿里云): 支持真正的流式输出，需要云服务账号
2. **vLLM-Omni** (计划中): 官方正在开发 Online Serving 支持

**参考**: [GitHub Issue #10](https://github.com/QwenLM/Qwen3-TTS/issues/10) - Qwen 团队官方回复

## Voice Cloning (声音克隆)

### 支持的模型

| 模型 | Voice Cloning | 参数 |
|------|---------------|------|
| GLM-TTS | ✅ Zero-shot | `prompt_wav`, `prompt_text` |
| CosyVoice | ✅ Zero-shot | `prompt_wav`, `prompt_text` |
| Qwen3-TTS (Base) | ✅ Zero-shot | `ref_audio`, `ref_text` |

### GLM-TTS Voice Cloning

使用 `prompt_wav` 和 `prompt_text` 参数：

```yaml
- name: "glm_tts_vc"
  type: "glm_tts"
  prompt_wav: "./processed_audio/normalized_Yang.wav"
  prompt_text: "Hello everyone, welcome back to CS294-137."
```

### CosyVoice Voice Cloning

使用 `zero_shot` 模式，通过 `prompt_wav` 和 `prompt_text` 参数：

```yaml
- name: "cosyvoice"
  type: "cosyvoice"
  mode: "zero_shot"
  prompt_wav: "./prompts/speaker.wav"
  prompt_text: "参考音频对应的文本内容"
```

### Qwen3-TTS Voice Cloning

**注意**：Voice Cloning 需要使用 **Base 模型**，不是 CustomVoice 模型。

```yaml
- name: "qwen_tts_vc"
  type: "qwen_tts_vc"
  model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
  ref_audio: "./prompts/speaker.wav"
  ref_text: "参考音频对应的文本"  # 可选，提供可提升质量
  x_vector_only: false  # true=仅用声纹，false=ICL模式
```

#### Qwen3-TTS API 文档 (qwen-tts 库)

**重要**: `generate_voice_clone()` 和 `generate_custom_voice()` 返回的是元组，不是直接的音频！

```python
from qwen_tts import Qwen3TTSModel

# 加载模型 (device 自动检测，不要手动传入)
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")

# Voice Cloning 合成
# 返回值: Tuple[List[numpy.ndarray], int] -> (音频列表, 采样率)
audio_list, sample_rate = model.generate_voice_clone(
    text="要合成的文本",
    ref_audio=(audio_np, ref_sr),  # (numpy数组, 采样率) 或文件路径
    ref_text="参考音频对应的文本",  # 可选，提供可提升质量
    x_vector_only_mode=False,       # True=仅用声纹，False=ICL模式
)
audio = audio_list[0]  # 获取第一个音频

# CustomVoice 合成 (预设说话人)
audio_list, sample_rate = model.generate_custom_voice(
    text="要合成的文本",
    speaker="Vivian",
    language="English",
)
audio = audio_list[0]

# 创建 Voice Clone Prompt (复用参考音频)
prompt = model.create_voice_clone_prompt(
    ref_audio=(audio_np, ref_sr),
    ref_text="参考文本",
)
```

**常见错误**:
- `'tuple' object has no attribute 'flatten'` → 忘记解包返回值，应使用 `audio_list, sr = model.generate_voice_clone(...)`
- `'Qwen3TTSModel' object has no attribute 'to'` → qwen-tts 自动管理设备，不需要 `.to(device)`
- `unexpected keyword argument 'device'` → `from_pretrained()` 不接受 `device` 参数

### 可用的参考音频

`processed_audio/` 目录下的预处理音频：

| Voice | Audio File | Transcript |
|-------|------------|------------|
| Yang | `normalized_Yang.wav` | "Hello everyone, welcome back to CS294-137. Today, we will continue our discussion in computer vision." |
| DeNero | `normalized_DeNero.wav` | "In this course, we've discussed trees many times." |

### 参考音频要求

- 格式：WAV（推荐）、MP3
- 时长：3-30 秒（推荐 5-15 秒）
- 质量：清晰、无背景噪音
- 内容：自然说话，避免唱歌或特殊音效

## 输出目录结构

```
outputs/
└── 20240201_143052_glm_tts-qwen_tts/
    ├── scripts.json          # 测试文本
    ├── metrics.json          # 汇总指标
    ├── detailed_results.json # 详细结果
    ├── report.md             # 对比报告（Markdown 表格）
    ├── glm_tts/
    │   ├── test_001.wav
    │   └── test_002.wav
    └── qwen_tts/
        ├── test_001.wav
        └── test_002.wav
```

## 依赖管理

### vLLM-Omni 加速 (可选)

vLLM-Omni 可显著提升 Qwen3-TTS 的 RTF 性能。

**注意**：vLLM 仅支持 **Linux**！Windows 用户需使用 WSL2。

#### WSL2 安装 (Windows 用户)

```bash
# 1. 打开 WSL
wsl

# 2. 进入项目目录 (Windows D: 盘 -> /mnt/d)
cd /mnt/d/tai/tts-evaluation

# 3. 安装 Miniconda (如未安装)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# 4. 安装 vLLM 环境
python setup_models.py --qwen-vllm

# 5. 验证
conda run -n tts-qwen-vllm python -c "from vllm import LLM; print('vLLM OK')"
```

#### Linux 直接安装

```bash
python setup_models.py --qwen-vllm
```

#### 从 WSL 运行 vLLM 评估

```bash
# 方法 1: 在 WSL 终端中
cd /mnt/d/tai/tts-evaluation
python main.py -m qwen_tts_vllm

# 方法 2: 从 Windows CMD/PowerShell 调用
wsl -e bash -c "cd /mnt/d/tai/tts-evaluation && python main.py -m qwen_tts_vllm"
```

#### 使用 vLLM 进行评估

在 `config.yaml` 中使用 `qwen_tts_vllm` 类型：

```yaml
models:
  # 标准 Qwen3-TTS（对比基准）
  - name: "qwen_tts"
    type: "qwen_tts"
    model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    speaker: "Vivian"

  # vLLM 加速版本
  - name: "qwen_tts_vllm"
    type: "qwen_tts_vllm"
    model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    speaker: "Vivian"
    # vLLM 特有参数
    temperature: 0.9
    top_p: 1.0
    top_k: 50
    max_tokens: 2048
    repetition_penalty: 1.05
```

#### vLLM Voice Cloning

```yaml
- name: "qwen_tts_vllm_vc"
  type: "qwen_tts_vllm_vc"
  model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
  ref_audio: "./prompts/speaker.wav"
  ref_text: "参考音频文本"
  mode: "icl"  # "icl" 或 "xvec_only"
```

#### 测试单请求性能

使用只有 1 个 sample 的 dataset：

```bash
# 创建单样本测试文件
echo '[{"id": "single_001", "text": "Hello world."}]' > scripts/single_test.json

# 运行
python main.py -m qwen_tts_vllm -s scripts/single_test.json
```

#### RTF 性能对比

| 后端 | RTF (典型值) | 优势 |
|------|-------------|------|
| `qwen-tts` 原生 | ~0.3-0.5 | 简单、稳定 |
| `vLLM-Omni` (batch) | ~0.1-0.2 | 更快，支持批量 |
| `vLLM-Omni` (single) | ~0.2-0.3 | 测量单请求开销 |

### CosyVoice vLLM 加速

**✅ 状态：可用** - 稳态 RTF ~0.2-0.4

CosyVoice vLLM 现在可以正常运行，稳态性能与普通 CosyVoice 相当。

```yaml
# 标准模式（streaming）
- name: "cosyvoice_vllm"
  type: "cosyvoice_vllm"
  enabled: false
  model_path: "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
  repo_path: "./CosyVoice"

# 单请求模式（non-streaming，更准确的 per-request 指标）
- name: "cosyvoice_vllm_single"
  type: "cosyvoice_vllm"
  enabled: false
  single_batch: true  # 禁用 streaming，每次请求独立测量
  model_path: "./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
  repo_path: "./CosyVoice"
```

**环境要求**：
- 使用独立环境 `tts-cosyvoice-vllm`
- 安装 conda gcc 编译器：`gcc_linux-64`, `gxx_linux-64`
- 安装 `onnxruntime-gpu`（关键！否则 RTF 会很差）
- ruamel.yaml 版本限制：`>=0.17.28,<0.19`

**性能特点**：
- 首次运行有 ~30 秒预热（CUDA Graph 编译 + prompt 处理）
- 稳态 RTF ~0.2-0.4（与普通 CosyVoice 相当）
- 批量处理时可能有优势

**WSL 注意事项**：
- vLLM 在 WSL 下使用 `pin_memory=False`
- 确保安装 `onnxruntime-gpu` 而非 `onnxruntime`

### 环境依赖对照

| 环境 | YAML 文件 | 依赖来源 | 平台 | 状态 |
|------|----------|---------|------|------|
| `tts-glm` | `envs/glm_tts.yaml` | GLM-TTS/requirements.txt | All | ✅ |
| `tts-qwen` | `envs/qwen_tts.yaml` | `qwen-tts` PyPI 包 | All | ✅ |
| `tts-qwen-vllm` | `envs/qwen_tts_vllm.yaml` | vLLM-Omni | Linux | ✅ |
| `tts-cosyvoice` | `envs/cosyvoice.yaml` | CosyVoice/requirements.txt | All | ✅ |
| `tts-cosyvoice-vllm` | `envs/cosyvoice_vllm.yaml` | vLLM + CosyVoice | Linux | ✅ |

### 关键依赖版本

**PyTorch 版本**: 所有环境统一使用 `torch==2.3.1` + `pytorch-cuda=12.1`

**GLM-TTS 特殊依赖**:
- `funasr==1.1.6` - 语音识别
- `WeTextProcessing==1.0.3` - 文本处理
- `x_transformers==1.42.26` - Transformer 实现

**Qwen-TTS 特殊依赖**:
- `qwen-tts>=0.0.5` - 官方包，自动处理模型加载
- `flash-attn` - 可选，需单独安装：`pip install flash-attn --no-build-isolation`

**CosyVoice 特殊依赖**:
- `pynini=2.1.5` - 必须通过 conda 安装（不是 pip）
- `conformer==0.3.2` - 语音模型
- `matcha-tts` - 通过 third_party 子模块提供

### 安装后验证

```bash
# 验证 GLM-TTS 环境
conda run -n tts-glm python -c "import torch; from transformers import AutoModel; print('GLM OK')"

# 验证 Qwen-TTS 环境
conda run -n tts-qwen python -c "from qwen_tts import Qwen3TTSModel; print('Qwen OK')"

# 验证 CosyVoice 环境
conda run -n tts-cosyvoice python -c "import torch; import pynini; print('CosyVoice OK')"

# 验证 vLLM 环境 (Linux only)
conda run -n tts-qwen-vllm python -c "from vllm import LLM; from vllm_omni import Omni; print('vLLM OK')"
```

### 常见问题

**Q: 安装 pynini 失败**
A: pynini 必须通过 conda 安装，不能用 pip。确保使用 conda-forge channel。

**Q: flash-attn 编译失败**
A: 需要 CUDA 开发工具。如果内存不足，使用 `MAX_JOBS=4 pip install flash-attn --no-build-isolation`

**Q: CosyVoice import 错误**
A: 确保 sys.path 包含 CosyVoice 仓库根目录和 `third_party/Matcha-TTS`

**Q: CUDA out of memory**
A: 各模型按顺序运行，每个模型推理完成后会自动释放 GPU 内存
