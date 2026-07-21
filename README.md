# DeepSeek-CAAFE + 普通 GP（Prompt 分离版）

该版本基于上传的 `jinhua-caafe-elite(13).py` 重构，核心算法和实验流程保持不变，主要完成以下拆分：

- `prompts/caafe_prompts.py`：System Prompt、用户 Prompt 和特征摘要构造。
- `prompts/feedback_prompts.py`：接受、拒绝、失败等下一轮反馈文本。
- `configs/dataset_descriptions.py`：数据集自然语言描述。
- `jinhua_caafe_elite_refactored.py`：数据处理、DeepSeek 调用、普通 GP 进化和实验主流程。

## 运行方法

1. 将 `stock.xlsx` 放在项目根目录，与主脚本同级。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 设置新的 DeepSeek API Key。不要把密钥写入代码。

PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="你的新密钥"
```

CMD：

```bat
set DEEPSEEK_API_KEY=你的新密钥
```

4. 运行：

```bash
python jinhua_caafe_elite_refactored.py
```

## 修改 Prompt

- 修改特征生成要求：编辑 `prompts/caafe_prompts.py`。
- 修改实验结果反馈：编辑 `prompts/feedback_prompts.py`。
- 修改数据集背景：编辑 `configs/dataset_descriptions.py`。

## 注意

- 原上传脚本中的 API Key 已从本版本删除。由于旧密钥曾写入文件，建议立即在平台后台撤销并重新创建。
- 当前 `build_model()` 仍保留原脚本设置，实际使用的是 `Ridge`，没有擅自改成随机森林。
- 当前 Prompt 固定每轮生成 3 个候选，因此 `N_CANDIDATES_PER_ROUND` 应保持为 3。
