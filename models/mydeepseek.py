import os
import numpy as np
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-84379869a1ed460d94eee8998bc4f3e5"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"


os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

client = OpenAI()


class MyDS:
    def __init__(self, model_name):
        self.model_name = model_name  #  The deepseek-chat model points to DeepSeek-V3. The deepseek-reasoner model points to DeepSeek-R1.
        self.client = OpenAI()


    def batch_chat(self, message, temperature=1.0, n=1):
        # print("====input_messages_list:", messages_list)

        open_ai_messages_list = []

        if isinstance(message[0], str):
            open_ai_messages_list.append(
                 {"role": "user", "content": message[0]}
                )
        else:
            open_ai_messages_list.append(
                {"role": "user", "content": message[0][0]}
            )

        # print("=====messages:=======",open_ai_messages_list)

        response = client.chat.completions.create(
            model= self.model_name,
            messages= open_ai_messages_list,
            max_tokens=1,
            stream=False,
            # temperature=1.3,
            logprobs=True,
            top_logprobs=5,
            n=n
        )


        if response.choices:
            first_choice = response.choices[0]

            if first_choice.logprobs and first_choice.logprobs.content:
                first_token_logprobs = first_choice.logprobs.content[0]


                logprob_yes = None
                logprob_no = None


                if first_token_logprobs.top_logprobs:
                    for top_logprob in first_token_logprobs.top_logprobs:
                        if top_logprob.token == "是":
                            logprob_yes = top_logprob.logprob
                        elif top_logprob.token == "否":
                            logprob_no = top_logprob.logprob


                logprob_yes = logprob_yes if logprob_yes is not None else float("-inf")
                logprob_no = logprob_no if logprob_no is not None else float("-inf")


                prob_yes = np.exp(logprob_yes) if logprob_yes > float("-inf") else 0
                prob_no = np.exp(logprob_no) if logprob_no > float("-inf") else 0


                denominator = prob_yes + prob_no
                s = prob_yes / denominator if denominator > 0 else 0.5


            else:
                print("No logprobs content found in the response.")
        else:
            print("No choices found in the response.")

        # print("-----res score", s)
        return s



    def batch_chat_multiclas(self, message, temperature=1.0, n=1):
        SYS = """你是一个专业的文本分类模型，需要对输入的新闻句子进行宣传手段的多分类任务。\
                                    *宣传手段类别包含 43 种，列表如下*：1. 诉诸人身。含义：攻击对方个人，而不是回应其论点，利用无关的个人特征来反驳对手或支持自身观点。2.议题设置。含义：大众媒体通过报道的方向和数量，对某一特定议题进行强调。3.美好人生。含义：通过与名人建立联系或描绘吸引人、幸福的人物形象来提升士气。4.戈培尔效应。含义：以微妙和间接的方式传递信息，使其被无意识接受，进而影响心理或行为反应。5.假二难推理。含义：在多种可能性存在的情况下，仅呈现有限的选项。6.单方论证。含义：只呈现支持性的理由，而忽略反对的观点。7.经典条件反射。含义：将两个经常一起出现的刺激联系起来，从而引发条件反射。8.认知失调。含义：持有两个相互矛盾的信念所产生的心理不适，导致个体改变行为或信念以减少紧张感。9.诉诸平民。含义：通过使用目标受众的语言和方式来赢得信任。10.个人崇拜。含义：通过大规模宣传，将某个人塑造成群体敬仰的对象。11. 妖魔化。含义：将敌人描绘成完全破坏性和邪恶的形象，通常涉及仇恨、个人攻击和诽谤。12.低落士气。含义：削弱对手的士气，以鼓励投降或叛变。13.独裁。含义：将某种观点呈现为唯一可行的选择，通过消除其他选项来简化决策过程。14.虚假信息。含义：篡改或删除公共记录，以编造关于事件、个人或组织的虚假叙述。15.分而治之。含义：将更大的力量拆分为较小的、力量较弱的部分，以防止统一反对。16.以退为进。含义：先提出一个极端的请求，使其被拒绝，然后再提出一个更合理的请求，以提高成功率。17.粗直语。含义：使用严厉或冒犯性的语言，负面地影响观众的感知。 18.委婉语。含义：使用温和或间接的表达方式来替代严厉或不愉快的语言。19.制造幸福感。含义：通过使用愉快或吸引人的事件来提升士气。20. 虚假指控。含义：提出没有事实依据的指控或主张。21.谎言灌喷。含义：通过多个渠道迅速且反复地传播大量信息，不顾真相或一致性。22.得寸进尺。含义：通过先提出一个小请求，再提出一个更大的请求来进行劝说。23.框架化。含义：通过以不同方式呈现相同的问题，使某个选项看起来更有利。24.煤气灯效应。含义：通过否认、虚假信息和矛盾来操控某人，让其怀疑自己的记忆、感知或理智。25. 乱枪打鸟。含义：通过大量的论点压倒对手，其中许多论点是有缺陷或不相关的。26.光辉普照。含义：将积极的概念与信息关联，使其在没有证据的情况下更具吸引力。27.关联谬误。含义：通过无关的关联，论证一种事物的特性适用于另一个事物。28.片面事实。含义：一个部分正确但遗漏关键信息的陈述。29.含糊其辞。含义：使用模糊或不明确的术语，进行无分析或无正当理由的操控。30.接受范围。含义：超出个人或群体接受范围的信息，通常会引发心理抵抗。31.负载性语言。含义：使用具有强烈内涵的词汇来激发情绪反应或利用刻板印象。32.谎言与欺骗。含义：使用虚假或扭曲的信息来为行动或信仰辩护，并促使其被接受。33.社会环境控制。含义：通过同伴或社会压力强制执行对某一理念或事业的遵从，与洗脑和精神控制相关。34.冷处理。含义：在明确否定不可能的情况下，最小化事件或情感的重要性。35.人身攻击。含义：批评或攻击一个人的个人特征，而不是他们的论点。36.形式谬误。含义：一个有缺陷的推理，结论没有逻辑地从前提中得出。37.混淆。含义：故意模糊不清的交流，旨在使观众困惑或排除更广泛的理解。38.操作制约。含义：通过奖励或惩罚影响‘自愿’表现的行为。39.：单音谬误。含义：将结果归因于单一原因，而忽视其他贡献因素。40.单一思想。含义：通过简化的论证压制替代观点。41.断章取义。含义：删除陈述的原始上下文，创造误导性的推论。42.暗示性语言。含义：间接传达观点、情感或含义，而不明确声明。43.终结性陈词。含义：一种总结性的语言策略，旨在强化信息并促使观众做出最终行动。"
                                     **任务要求**： 1. 输入是一个新闻句子，可能包含 0 到多种宣传手段。2.输出只能包含该句子涉及的所有宣传手段的 *类别编号*，不包含任何解释，多个类别用英文逗号 `,` 分隔; 3. 如果该句子不包含任何宣传手段，请返回 `0`。 \
                                    给定新闻:"""

        QUS = """\n 请返回新闻中所有可能存在的宣传手段类别编号，不同类别用 `,` 分隔:"""
        open_ai_messages_list = []

        if isinstance(message[0], str):
            mes =  SYS + message[0] +  QUS
            open_ai_messages_list.append(
                 {"role": "user", "content": mes}
                )
        else:
            mes = SYS + message[0][0] + QUS
            open_ai_messages_list.append(
                {"role": "user", "content": mes}
            )

        # print("=====输给ds的messages:=======",open_ai_messages_list)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=open_ai_messages_list,
            max_tokens=20,
            stream=False,
            logprobs=True,
            # top_logprobs=5
        )

        propaganda_scores = {}

        if response and hasattr(response, "choices") and response.choices:
            for token_info in response.choices[0].logprobs.content:
                prob = np.exp(token_info.logprob)
                token_text = token_info.token.strip()

                if token_text.isdigit():
                    propaganda_scores[int(token_text)] = prob

        # print("====propaganda_scores-=====", propaganda_scores)
        # return propaganda_scores

        else_propaganda_mapping = {
            'P7': '诉诸人身', 'P9': '议题设置', 'P17': '美好人生', 'P18': '戈培尔效应',
            'P19': '假二难推理', 'P20': '单方论证', 'P21': '经典条件反射', 'P22': '认知失调',
            'P23': '诉诸平民', 'P24': '个人崇拜', 'P25': '铁魔化', 'P26': '低落士气',
            'P27': '独裁', 'P28': '虚假信息', 'P29': '分而治之', 'P30': '以退为进',
            'P31': '粗直语', 'P32': '委婉语', 'P33': '制造幸福感', 'P35': '虚假指控',
            'P37': '谎言灌喷', 'P38': '得寸进尺', 'P39': '框架化', 'P40': '煤气灯效应',
            'P41': '乱枪打鸟', 'P42': '光辉普照', 'P43': '关联谬误', 'P44': '片面事实',
            'P45': '含糊其辞', 'P46': '接受范围', 'P47': '负载性语言', 'P48': '谎言与欺骗',
            'P49': '社会环境控制', 'P50': '冷处理', 'P51': '人身攻击', 'P52': '形式谬误',
            'P53': '混淆', 'P54': '操作制约', 'P55': '单音谬误', 'P56': '单一思想',
            'P57': '断章取义', 'P58': '暗示性语言', 'P59': '终结性陈词'
        }

        p7_exp_prob_mapping = {}
        mapping_keys = list(else_propaganda_mapping.keys())

        for i, key in enumerate(mapping_keys):
            index = i + 1
            p7_exp_prob_mapping[key] = propaganda_scores.get(index, 0.0)

        # print("====p7_exp_prob_mapping=====", p7_exp_prob_mapping)

        return p7_exp_prob_mapping





