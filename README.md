# 使用GPT-FINETUNE与GPT-INDEX构建自定义问答机器人

本文档将指导您完成以下操作：

1. 准备数据集
2. 训练模型
3. 使用训练好的模型
4. 使用带前端的聊天机器人

**环境要求：**

- 操作系统：macOS 13.2.1 (22D68)
- Python 版本：3.9.6

**所需库及工具：**

- openai
- gradio
- gpt_index
- langchain
- logging
- Jupyter 编辑器

使用以下命令安装所需库：

```bash
pip install openai gradio gpt_index langchain logging
```

代码结构
- GPT_3_Finetuning_ipynb.ipynb
- README.md
- content
    - atis_intents.csv
- data
    - atis_intents.csv
    - atis_intents_test.csv
    - atis_intents_train.csv
    - sample_data.csv
- doc.py
- flagged
- flight.py
- graham_essay
    - DavinciComparison.ipynb
    - InsertDemo.ipynb
    - KeywordTableComparison.ipynb
    - TestEssay.ipynb
    - data
        - paul_graham_essay.txt
    - index.json
    - index_list.json
    - index_table.json
    - index_tree_insert.json
    - index_with_query.json
- index.json
- intent_sample.jsonl
- trainPerson.iml

**GPT_3_Finetuning_ipynb.ipynb: 包含了 GPT-3 模型的 finetuning 和测试代码的 Jupyter Notebook 文件。
README.md: 项目的说明文档。
content/atis_intents.csv: 原始的 ATIS 数据集。
data/atis_intents.csv: 经过处理后的 ATIS 数据集。
data/atis_intents_train.csv: ATIS 数据集中的训练集。
data/atis_intents_test.csv: ATIS 数据集中的测试集。
data/sample_data.csv: 处理后的 ATIS 数据集的一个子集，用于模型 finetuning。
doc.py: 基于 GPT 模型的文档智能问答机器人的代码。
flagged/: 存储被标记的数据集文件。
flight.py: 基于 GPT 模型的航班智能问答机器人的代码。
graham_essay/: 存储 Paul Graham 文章相关的代码和数据。
index.json: 存储了 GPT 索引的中间结果。
intent_sample.jsonl: ATIS 数据集的一个子集，用于模型 finetuning。
trainPerson.iml: IntelliJ IDEA 的项目文件。**

## 背景

目标是创建一个自定义问答机器人。为实现这个目标，我们可以使用两种方法：使用使用官方定制服务finetune和gpt-index。

## 第一种
## 1. 准备数据集

在开始训练模型之前，我们需要准备数据集。以下是准备数据集的步骤：

### 1.1 清洗数据

数据清洗是数据预处理的重要部分，我们需要确保数据质量以获得更好的训练效果。

### 1.2 示例：航班问题

这里，我们以航班问题为例，展示清洗后的原始数据。

## 2. 开始训练

训练结束后，处理数据。

## 3. 使用训练好的模型

设置 对应训练账户的API Key：

```python
os.environ['OPENAI_API_KEY'] = "sk-***"
```
调用训练好的模型进行预测：
``
```
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
model="davinci:ft-personal-2023-04-03-04-12-25",
prompt="Do we have london flight on Monday?",
temperature=0.7,
max_tokens=256,
top_p=1,
frequency_penalty=0,
presence_penalty=0,
stop=["END"]
)
print(response)
```


## 4. 
- 使用训练好的模型
`python flight.py`

模型名称：davinci:ft-personal-2023-04-03—04-15-25

- 如何训练
``

这个代码块展示了如何使用 ATIS Airline Travel Information System 数据集对 GPT 进行微调。首先，需要安装 openai 库并设置 API Key。之后，按照以下步骤处理数据集：

1.读取数据并设置列名

2. 提取有关航班、地面服务、票价、缩写和飞行时间的意图

3.对每个意图采样40条数据

4.将数据处理成 JSONL 格式

5.使用 openai 库准备数据

6.使用 openai 库微调模型

7.跟踪微调进度
使用微调后的模型进行预测

8.训练好的模型会在十几分钟后显示在playground上。
可以在上面直接运行 调整参数测试。

9。
补充:
代码见GPT_3_Finetuning_ipynb.ipynb. 使用Jupyter编辑器运行代码。


这一行:修改输入数据路径

data = pd.read_csv("./content/atis_intents.csv",header=None)

本例中运行截止到，后面的根据需要运行。

sample_data.to_json("intent_sample.jsonl", orient='records', lines=True)

运行之前将API Key设置为os.environ['OPENAI_API_KEY'] = ""。

之后，代码从ATIS数据集中读取数据，并对数据进行预处理，以便微调GPT。

在处理数据之后，它将数据分为训练集和验证集，并使用openai库将数据准备好。最后，它使用openai库微调模型，并跟踪微调进度。

在微调完成之后，您可以使用微调后的模型进行预测。

在航班例子中，它使用了openai.Completion.create()方法来进行预测。  
``




## 5. 带前端的聊天机器人
- 5.1 访问地址

-- 本地 URL：http://127.0.0.1:7860

--  公共 URL：https://728685f5c04be13e3a.gradio.live
- 5.2 前端效果图
![效果](https://p.ipic.vip/ywmyam.png)



## 第二种
基本流程类似，
区别在于使用了[llama_index](https://github.com/jerryjliu/llama_index)。
增加了数据库支持，处理数据更加直接，而且可以自定义模型进行数据处理。
运行效果一样见上图。

代码参考:
```

def construct_index():
    # define LLM
    documents = SimpleDirectoryReader('./graham_essay/data').load_data()
    index = GPTSimpleVectorIndex.from_documents(documents)
    # save
    index.save_to_disk("index.json")
    return index


def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response


iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Doc AI Chatbot")

index = construct_index()
iface.launch(share=True)

```

**代码解释**

1.实现了一个基于 GPT 模型的文档智能问答机器人，能够根据用户输入的问题自动给出相应的回答。

2.代码中使用了 gpt_index 库，其中 construct_index() 函数定义了 LLM 模型的构建过程，即将指定目录下的文档进行处理，将其转换成向量表示，并构建一个简单的索引，然后保存到磁盘上。

3.chatbot() 函数实现了问答逻辑，即根据用户输入的问题在构建好的索引中查询相似的文档，并返回相应的回答。

4.gr.Interface() 函数实现了一个简单的 Web 前端，让用户可以通过文本框输入问题，并获取机器人的回答。

5.用户的输入会被 chatbot() 函数处理，并返回一个字符串作为输出。


## 补充: 模型对比 
- GPT-3
- GPT-3.5
- PT-4
- Chinese-LLaMA-Alpaca
- stanford_alpaca
- gpt4All

相关研究链接:

[gpt-neo](https://github.com/EleutherAI/gpt-neo/)

[gpt的复现之路](https://wqw547243068.github.io/chatgpt_mimic)

[达摩院gpt3莫塔社区](https://www.modelscope.cn/models/damo/nlp_gpt3_text-generation_2.7B/summary)

[Stanford Alpaca  ChatGPT 学术版开源实现](https://zhuanlan.zhihu.com/p/614354549)

[GPT公开复制为什么失败，我们应该如何使用GPT-3.5/ChatGPT](https://hub.baai.ac.cn/view/24224)
