# 特殊疑问词
Special_Question_Prefix = ['what', 'why', 'when', 'where', 'who', 'how', 'which', 'whose']
# 一般性疑问词
General_Question_Prefix = ['do', 'does', 'did', 'am', 'is', 'are', 'were', 'was', 'can', 'could', 'will', 'would',
                           'have', 'has', 'had', 'should', 'shall']

# Take P1 as an example, the corresponding general question is:
# "$EVIDENCE$\nMessage: $MESSAGE$\n"+"Is the message is LABEL?"
# Init_Predicate = {'Special': {
#     'VENUE': "What is the source of the information?",
#     'POSTTIME': "When was it published?",
#     'PUBLISHER': "Who did publish it? ",
#     'REPUTATION': "How about the background and reputation of the publisher?",
#     'INTENT': "What is the intent of this message?",
#     'STATEMENTS': "What are the important statements of this message for verification?",
#     'EVIDENCES': "What are relevant evidences to predict the veracity label of STATEMENT",
#     'EVIDENCE': "What are relevant evidences to predict the veracity label of MESSAGE"
# },
#
#     'General': {
#         'P1': ["Is the message is true?", [("Background information", "EVIDENCE"), ("Message", "MESSAGE")]],
#         'P2': ["Did the message contain adequate background information?", [("消息", "MESSAGE")]],
#         'P3': ["Is the background information in message accurate and objective?", [("消息", "MESSAGE")]],
#         'P4': [
#             "Is there any content in message that has been intentionally eliminated with the meaning being distorted?",
#             [("消息", "MESSAGE")]],
#         'P5': ["Is there an improper intention (political motive, commercial purpose, etc.) in the message?",
#                [("消息", "MESSAGE"), ("Intent", "INTENT")]],
#         'P6': ["Does the publisher have a history of publishing information with improper intention?",
#                [("Publisher Reputation", "REPUTATION")]],
#         'P7': ["Is the statement is true?", [("Background information", "EVIDENCES"), ("Statement", "STATEMENTS")]],
#     },
#     'G_prefix': "Evidence:EVIDENCE\n\nMessage: CLAIM\n\n",
#     'Q': "Q: "
# }

# P2,P3,P4  Retrieving the contextual information
# P4<-P5,P6  Is there any content that has been intentionally eliminated with the meaning being distorted?
# P7<-fact checking

# Prompt_IE_FC = refer to the markdown file

# 初始化问题模板
# Special 部分存储了与消息相关的基础性问题
Init_Predicate = {'Special': {
    'VENUE': "What is the source of the information?",
    'POSTTIME': "When was it published?",
    'PUBLISHER': "Who did publish it? ",
    'REPUTATION': "How about the background and reputation of the publisher?",
    'INTENT': "What is the intent of this message?",
    'STATEMENTS': "What are the important statements of this message for verification?",
    'EVIDENCES': "What are relevant evidences to predict the veracity label of STATEMENT",
    'EVIDENCE': "What are relevant evidences to predict the veracity label of MESSAGE"
},
#     # General 部分包含了通用问题模板
# 'General': {
#         # each sample in dataset is a dict, EVIDENCE/Message (the second element of the tuple) is the key of the dict.
#         'P1': ["Does the message use phrases like 'in hindsight, it's obvious' to suggest predictable outcomes ?", [("Message", "MESSAGE")]],
#         'P2': ["Does the message judge decisions based on outcomes, ignoring available information at the time ?", [("Message", "MESSAGE")]],
#         # P2 '简化因果'
#         'P3': ["Does the message use exaggerated or extreme language to amplify the event's significance ?", [("Message", "MESSAGE")]],
#         # P3 '夸张（夸大或淡化）'
#         'P4': ["Does the message claim a decision was 'bound to succeed' or 'bound to fail' ?", [("Message", "MESSAGE")]],
#         # P4 '加载语言（情绪语言）'
#         'P5': ["Does the message selectively focus on historical events that support its viewpoint ?", [("Message", "MESSAGE")]],
#         'P6': ["Does the message imply that different choices would have led to entirely different outcomes ?", [("Message", "MESSAGE")]],
#         'P7': ["Does the message emphasize the role of chance and external factors in outcomes ?", [("Message", "MESSAGE")]],
#         'P8': ["Does the message judge decisions based on outcomes, ignoring available information at the time Does the sentence explore multiple perspectives, suggesting various possible results and causes ?", [("Message", "MESSAGE")]],
#         },
#     # 用于格式化问题
#     # Evidence: [证据文本]
#     # Message: [声明文本]
#     # Q: [问题]
#     'G_prefix': "Evidence:EVIDENCE\n\nMessage: CLAIM\n\n",
#     'Q': "Q: "
# }
    # 根据我们的cognitive数据集定义谓词模板
    'General': {
    'P1': [
    "给定一种宣传手法：情绪语言。含义：使用具有强烈情感含义（正面或负面）的特定词语和短语来影响受众。该消息是否使用了这种宣传手法？",
    [("消息", "MESSAGE")]],  # 情绪语言
    'P2': [
    "给定一种宣传手法：贴标签。含义：将宣传活动的对象标记为目标受众所恐惧、厌恶，或相反地，钦佩或赞扬的事物。该消息是否使用了贴标签的宣传手法？",
    [("消息", "MESSAGE")]],  # 贴标签
'P3': [
    "给定一种宣传手法：挥舞旗帜。含义：利用强烈的民族主义情绪（例如种族、性别、政治偏好）来为行动或理念辩护或宣传，通常强调特定的价值观或愿景。该消息是否使用了挥舞旗帜的宣传手法？",
    [("消息", "MESSAGE")]],  # 挥舞旗帜
'P4': [
    "给定一种宣传手法：简化因果。含义：假设某个问题只有单一原因，而实际上可能有多个原因，包括替罪羊策略，即在未深入调查问题复杂性的情况下，将责任推卸给个人或群体。该消息是否使用了简化因果的宣传手法？",
    [("消息", "MESSAGE")]],  # 简化因果
'P5': [
    "给定一种宣传手法：喊口号。含义：使用简短、朗朗上口的短语，可能包含标签化和刻板印象。该消息是否使用了喊口号的宣传手法？",
    [("消息", "MESSAGE")]],  # 喊口号
'P6': [
    "给定一种宣传手法：非黑即白。含义：将两种选择作为唯一可能性呈现，而实际上可能存在更多选项。该消息是否使用了非黑即白的宣传手法？",
    [("消息", "MESSAGE")]],  # 非黑即白
'P7': [
    "给定一种宣传手法：诉诸人身。含义：攻击对方个人，而不是回应其论点，利用无关的个人特征来反驳对手或支持自身观点。该消息是否使用了诉诸人身的宣传手法？",
    [("消息", "MESSAGE")]],  # 诉诸人身
'P8': [
    "给定一种宣传手法：诉诸反复。含义：通过重复强调某观点已经被充分讨论或反对意见已被驳斥，从而避免提供证据。该消息是否使用了诉诸反复的宣传手法？",
    [("消息", "MESSAGE")]],  # 诉诸反复
'P9': [
    "给定一种宣传手法：议题设置。含义：大众媒体通过报道的方向和数量，对某一特定议题进行强调。该消息是否使用了议题设置的宣传手法？",
    [("消息", "MESSAGE")]],  # 议题设置
'P10': [
    "给定一种宣传手法：诉诸权威。含义：仅仅因为某个权威或专家支持某一观点，就宣称该观点为真，而不提供其他证据。该消息是否使用了诉诸权威的宣传手法？",
    [("消息", "MESSAGE")]],  # 诉诸权威
'P11': [
    "给定一种宣传手法：诉诸恐惧/偏见。含义：通过激发对替代方案的焦虑或恐慌，基于先入为主的判断来寻求支持。该消息是否使用了诉诸恐惧/偏见的宣传手法？",
    [("消息", "MESSAGE")]],  # 诉诸恐惧/偏见
'P12': [
    "给定一种宣传手法：格言论证。含义：使用特定的词句，使得对该话题的批判性思考和有意义讨论变得困难。该消息是否使用了格言论证的宣传手法？",
    [("消息", "MESSAGE")]],  # 格言论证
'P13': [
    "给定一种宣传手法：诉诸转移。含义：引入无关内容，以转移对核心问题的注意力。该消息是否使用了诉诸转移的宣传手法？",
    [("消息", "MESSAGE")]],  # 诉诸转移
'P14': [
    "给定一种宣传手段：历史引用。含义：使用历史事件或人物来类比当前事件。该信息是否使用了历史引用的宣传手法？",
    [("消息", "MESSAGE")]],  # 引用历史
'P15': [
    "给定一种宣传手段：预设立场。含义：使用修辞标记来表达媒体的预设立场。该信息是否使用了预设立场的宣传手法？",
    [("消息", "MESSAGE")]],  # 预设立场
'P16': [
    "给定一种宣传手段：诉诸潮流。含义：为了避免社会孤立而倾向于与多数人保持一致。该信息是否使用了诉诸潮流的宣传手法？",
    [("消息", "MESSAGE")]],  # 诉诸潮流
'P17': [
    "给定一种宣传手段：美好人生。含义：通过与名人建立联系或描绘吸引人、幸福的人物形象来提升士气。该信息是否使用了美好人生的宣传手法？",
    [("消息", "MESSAGE")]],  # 美好人生
'P18': [
    "给定一种宣传手段：戈培尔效应。含义：以微妙和间接的方式传递信息，使其被无意识接受，进而影响心理或行为反应。该信息是否使用了戈培尔效应的宣传手法？",
    [("消息", "MESSAGE")]],  # 戈培尔效应
'P19': [
    "给定一种宣传手段：假二难推理。含义：在多种可能性存在的情况下，仅呈现有限的选项。该信息是否使用了假二难推理的宣传手法？",
    [("消息", "MESSAGE")]],  # 假二难推理
'P20': [
    "给定一种宣传手段：单方论证。含义：只呈现支持性的理由，而忽略反对的观点。该信息是否使用了单方论证的宣传手法？",
    [("消息", "MESSAGE")]],  # 单方论证
'P21': [
    "给定一种宣传手段：经典条件反射。含义：将两个经常一起出现的刺激联系起来，从而引发条件反射。该信息是否使用了经典条件反射的宣传手法？",
    [("消息", "MESSAGE")]],  # 经典条件反射
'P22': [
    "给定一种宣传手段：认知失调。含义：持有两个相互矛盾的信念所产生的心理不适，导致个体改变行为或信念以减少紧张感。该信息是否使用了认知失调的宣传手法？",
    [("消息", "MESSAGE")]],  # 认知失调
'P23': [
    "给定一种宣传手段：诉诸平民。含义：通过使用目标受众的语言和方式来赢得信任。该信息是否使用了诉诸平民的宣传手法？",
    [("消息", "MESSAGE")]],  # 诉诸平民
'P24': [
    "给定一种宣传手段：个人崇拜。含义：通过大规模宣传，将某个人塑造成群体敬仰的对象。该信息是否使用了个人崇拜的宣传手法？",
    [("消息", "MESSAGE")]],  # 个人崇拜
'P25': [
    "给定一种宣传手段：妖魔化。含义：将敌人描绘成完全破坏性和邪恶的形象，通常涉及仇恨、个人攻击和诽谤。该信息是否使用了妖魔化的宣传手法？",
    [("消息", "MESSAGE")]],  # 妖魔化
'P26': [
    "给定一种宣传手段：低落士气。含义：削弱对手的士气，以鼓励投降或叛变。该信息是否使用了低落士气的宣传手法？",
    [("消息", "MESSAGE")]],  # 低落士气
'P27': [
    "给定一种宣传手段：独裁。含义：将某种观点呈现为唯一可行的选择，通过消除其他选项来简化决策过程。该信息是否使用了独裁的宣传手法？",
    [("消息", "MESSAGE")]],  # 独裁
'P28': [
    "给定一种宣传手段：虚假信息。含义：篡改或删除公共记录，以编造关于事件、个人或组织的虚假叙述。该信息是否使用了虚假信息的宣传手法？",
    [("消息", "MESSAGE")]],  # 虚假信息
'P29': [
    "给定一种宣传手段：分而治之。含义：将更大的力量拆分为较小的、力量较弱的部分，以防止统一反对。该信息是否使用了分而治之的宣传手法？",
    [("消息", "MESSAGE")]],  # 分而治之
'P30': [
    "给定一种宣传手段：以退为进。含义：先提出一个极端的请求，使其被拒绝，然后再提出一个更合理的请求，以提高成功率。该信息是否使用了以退为进的宣传手法？",
    [("消息", "MESSAGE")]],  # 以退为进
'P31': [
            "给定一种宣传手段：粗直语。含义：使用严厉或冒犯性的语言，负面地影响观众的感知。该信息是否使用了粗直语的宣传手法？",
            [("消息", "MESSAGE")]],  # 粗直语
'P32': [
            "给定一种宣传手段：委婉语。含义：使用温和或间接的表达方式来替代严厉或不愉快的语言。该信息是否使用了委婉语的宣传手法？",
            [("消息", "MESSAGE")]],  # 委婉语
'P33': [
            "给定一种宣传手段：制造幸福感。含义：通过使用愉快或吸引人的事件来提升士气。该信息是否使用了制造幸福感的宣传手法？",
            [("消息", "MESSAGE")]],  # 制造幸福感
'P34': [
            "给定一种宣传手段：夸大。含义：过度强调或夸大事件或情况。该信息是否使用了夸大的宣传手法？",
            [("消息", "MESSAGE")]],  # 夸大
'P35': [
            "给定一种宣传手段：虚假指控。含义：提出没有事实依据的指控或主张。该信息是否使用了虚假指控的宣传手法？",
            [("消息", "MESSAGE")]],  # 虚假指控
'P36': [
            "给定一种宣传手段：恐惧、不确定与怀疑（FUD）。含义：传播负面、怀疑或虚假的信息来影响感知。该信息是否使用了恐惧、不确定与怀疑（FUD）的宣传手法？",
            [("消息", "MESSAGE")]],  # 恐惧、不确定与怀疑
'P37': [
            "给定一种宣传手段：谎言灌喷。含义：通过多个渠道迅速且反复地传播大量信息，不顾真相或一致性。该信息是否使用了谎言灌喷的宣传手法？",
            [("消息", "MESSAGE")]],  # 谎言灌喷
'P38': [
            "给定一种宣传手段：得寸进尺。含义：通过先提出一个小请求，再提出一个更大的请求来进行劝说。该信息是否使用了得寸进尺的宣传手法？",
            [("消息", "MESSAGE")]],  # 得寸进尺
'P39': [
            "给定一种宣传手段：框架化。含义：通过以不同方式呈现相同的问题，使某个选项看起来更有利。该信息是否使用了框架化的宣传手法？",
            [("消息", "MESSAGE")]],  # 框架化
'P40': [
            "给定一种宣传手段：煤气灯效应。含义：通过否认、虚假信息和矛盾来操控某人，让其怀疑自己的记忆、感知或理智。该信息是否使用了煤气灯效应的宣传手法？",
            [("消息", "MESSAGE")]],  # 煤气灯效应
'P41': [
            "给定一种宣传手段：乱枪打鸟。含义：通过大量的论点压倒对手，其中许多论点是有缺陷或不相关的。该信息是否使用了乱枪打鸟的宣传手法？",
            [("消息", "MESSAGE")]],  # 乱枪打鸟
'P42': [
            "给定一种宣传手段：光辉普照。含义：将积极的概念与信息关联，使其在没有证据的情况下更具吸引力。该信息是否使用了光辉普照的宣传手法？",
            [("消息", "MESSAGE")]],  # 光辉普照
'P43': [
            "给定一种宣传手段：关联谬误。含义：通过无关的关联，论证一种事物的特性适用于另一个事物。该信息是否使用了关联谬误的宣传手法？",
            [("消息", "MESSAGE")]],  # 关联谬误
'P44': [
            "给定一种宣传手段：片面事实。含义：一个部分正确但遗漏关键信息的陈述。该信息是否使用了片面事实的宣传手法？",
            [("消息", "MESSAGE")]],  # 片面事实
'P45': [
            "给定一种宣传手段：含糊其辞。含义：使用模糊或不明确的术语，进行无分析或无正当理由的操控。该信息是否使用了含糊其辞的宣传手法？",
            [("消息", "MESSAGE")]],  # 含糊其辞
'P46': [
            "给定一种宣传手段：接受范围。含义：超出个人或群体接受范围的信息，通常会引发心理抵抗。该信息是否使用了接受范围的宣传手法？",
            [("消息", "MESSAGE")]],  # 接受范围
'P47': [
            "给定一种宣传手段：负载性语言。含义：使用具有强烈内涵的词汇来激发情绪反应或利用刻板印象。该信息是否使用了负载性语言的宣传手法？",
            [("消息", "MESSAGE")]],  # 负载性语言
'P48': [
            "给定一种宣传手段：谎言与欺骗。含义：使用虚假或扭曲的信息来为行动或信仰辩护，并促使其被接受。该信息是否使用了谎言与欺骗的宣传手法？",
            [("消息", "MESSAGE")]],  # 谎言与欺骗
'P49': [
            "给定一种宣传手段：社会环境控制。含义：通过同伴或社会压力强制执行对某一理念或事业的遵从，与洗脑和精神控制相关。该信息是否使用了社会环境控制的宣传手法？",
            [("消息", "MESSAGE")]],  # 社会环境控制
'P50': [
            "给定一种宣传手段：冷处理。含义：在明确否定不可能的情况下，最小化事件或情感的重要性。该信息是否使用了冷处理的宣传手法？",
            [("消息", "MESSAGE")]],  # 冷处理
'P51': [
            "给定一种宣传手段：人身攻击。含义：批评或攻击一个人的个人特征，而不是他们的论点。该信息是否使用了人身攻击的宣传手法？",
            [("消息", "MESSAGE")]],  # 人身攻击
'P52': [
            "给定一种宣传手段：形式谬误。含义：一个有缺陷的推理，结论没有逻辑地从前提中得出。该信息是否使用了形式谬误的宣传手法？",
            [("消息", "MESSAGE")]],  # 形式谬误
'P53': [
            "给定一种宣传手段：混淆。含义：故意模糊不清的交流，旨在使观众困惑或排除更广泛的理解。该信息是否使用了混淆的宣传手法？",
            [("消息", "MESSAGE")]],  # 混淆
'P54': [
            "给定一种宣传手段：操作制约。含义：通过奖励或惩罚影响‘自愿’表现的行为。该信息是否使用了操作制约的宣传手法？",
            [("消息", "MESSAGE")]],  # 操作制约
'P55': [
            "给定一种宣传手段：单音谬误。含义：将结果归因于单一原因，而忽视其他贡献因素。该信息是否使用了单音谬误的宣传手法？",
            [("消息", "MESSAGE")]],  # 单音谬误
'P56': [
            "给定一种宣传手段：单一思想。含义：通过简化的论证压制替代观点。该信息是否使用了单一思想的宣传手法？",
            [("消息", "MESSAGE")]],  # 单一思想
'P57': [
            "给定一种宣传手段：断章取义。含义：删除陈述的原始上下文，创造误导性的推论。该信息是否使用了断章取义的宣传手法？",
            [("消息", "MESSAGE")]],  # 断章取义
'P58': [
            "给定一种宣传手段：暗示性语言。含义：间接传达观点、情感或含义，而不明确声明。该信息是否使用了暗示性语言的宣传手法？",
            [("消息", "MESSAGE")]],  # 暗示性语言
'P59': [
            "给定一种宣传手段：终结性陈词。含义：一种总结性的语言策略，旨在强化信息并促使观众做出最终行动。该信息是否使用了终结性陈词的宣传手法？",
            [("消息", "MESSAGE")]],  # 终结性陈词
'P60': ["消息是否反映乐观/悲观倾向或褒贬态度？", [("消息", "MESSAGE")]],  # 乐观/悲观倾向或褒贬态度
'P61': ["消息是否使用问题引导思考或命令促使行动？", [("消息", "MESSAGE")]],  # 问题引导思考或命令促使行动
'P62': ["消息是否通过幽默或讽刺制造情感冲突或共鸣？", [("消息", "MESSAGE")]],  # 幽默或讽刺制造情感冲突或共鸣
'P63': ["消息是否使用确定或不确定的语言？", [("消息", "MESSAGE")]],  # 确定或不确定的语言
'P64': ["消息是否使用统计数据？", [("消息", "MESSAGE")]],  # 统计数据
'P65': ["消息是否模糊信息来源？", [("消息", "MESSAGE")]],  # 模糊信息来源
'P66': ["消息是否使用对比结构？", [("消息", "MESSAGE")]],  # 对比结构
'P67': ["消息是否强调信息的频繁出现？", [("消息", "MESSAGE")]],  # 强调信息的频繁出现
'P68': ["消息是否增强品牌、观点或事件的熟悉度？", [("消息", "MESSAGE")]],  # 增强品牌、观点或事件的熟悉度
'P69': ["消息是否推断事件发生后是‘不可避免’的？", [("消息", "MESSAGE")]],  # 推断事件发生后是“不可避免”的
'P70': ["消息是否从单一特征推断其他属性？", [("消息", "MESSAGE")]],  # 从单一特征推断其他属性
'P71': ["消息是否强调损失或风险？", [("消息", "MESSAGE")]],  # 强调损失或风险
'P72': ["消息是否淡化收益或不确定性？", [("消息", "MESSAGE")]],  # 淡化收益或不确定性
'P73': ["消息是否对比风险和收益的表达？", [("消息", "MESSAGE")]],  # 对比风险和收益的表达
'P74': ["消息是否描述群体行为？", [("消息", "MESSAGE")]],  # 描述群体行为
'P75': ["消息是否区分‘我们与他们’？", [("消息", "MESSAGE")]],  # 区分“我们与他们”
'P76': ["消息是否使用群体压力？", [("消息", "MESSAGE")]],  # 使用群体压力
'P77': ["消息是否做出绝对陈述？", [("消息", "MESSAGE")]],  # 做出绝对陈述
'P78': ["消息是否模糊概率？", [("消息", "MESSAGE")]],  # 模糊概率
'P79': ["消息是否使用行话或行业特定术语？", [("消息", "MESSAGE")]],  # 使用行话或行业特定术语
'P80': ["消息是否使用隐喻、拟人或视觉隐喻？", [("消息", "MESSAGE")]],  # 使用隐喻、拟人或视觉隐喻
'P81': ["消息是否使用押韵或有节奏的语言？", [("消息", "MESSAGE")]],  # 使用押韵或有节奏的语言
'P82': ["消息是否使用被动语态？", [("消息", "MESSAGE")]],  # 使用被动语态
'P83': ["消息是否使用过去完成时来暗示确定性？", [("消息", "MESSAGE")]],  # 使用过去完成时来暗示确定性
'P84': ["消息是否预测未来事件？", [("消息", "MESSAGE")]],  # 预测未来事件
'P85': ["消息是否使用连接词来改变预期？", [("消息", "MESSAGE")]],  # 使用连接词来改变预期
'P86': ["消息是否将个别案例归纳为群体特征？", [("消息", "MESSAGE")]],  # 将个别案例归纳为群体特征
'P87': ["消息是否忽视多样性？", [("消息", "MESSAGE")]],  # 忽视多样性
'P88': ["消息是否忽视反论点或反例？", [("消息", "MESSAGE")]],  # 忽视反论点或反例
'P89': ["消息是否使用可得性偏差？", [("消息", "MESSAGE")]],  # 使用可得性偏差
'P90': ["消息是否通过将时间与结果联系起来误导？", [("消息", "MESSAGE")]],  # 通过将时间与结果联系起来误导
'P91': ["消息是否关注结果而忽视过程？", [("消息", "MESSAGE")]],  # 关注结果而忽视过程
'P92': ["消息是否事后添加结果以增强预见感？", [("消息", "MESSAGE")]],  # 事后添加结果以增强预见感
'P93': ["消息是否强调紧迫性？", [("消息", "MESSAGE")]],  # 强调紧迫性
'P94': ["背景信息是否使用了‘问题-解决结构’的叙事结构？", [("消息", "MESSAGE")]], # 强调紧迫性
'P95': ["背景信息是否使用了‘对比结构’的叙事结构？", [("消息", "MESSAGE")]], # 强调紧迫性
'P96': ["背景信息是否使用了‘英雄之旅结构’的叙事结构？", [("消息", "MESSAGE")]], # 强调紧迫性
'P97': ["背景信息是否使用了‘三幕剧结构’的叙事结构？", [("消息", "MESSAGE")]], # 强调紧迫性
'P98': ["背景信息是否使用了‘框架叙事结构’的叙事结构？", [("消息", "MESSAGE")]], # 强调紧迫性
'P99': ["背景信息是否使用了‘循环叙事结构’的叙事结构？", [("消息", "MESSAGE")]], # 强调紧迫性
'P100': ["背景信息是否使用了‘嵌套结构’的叙事结构？", [("消息", "MESSAGE")]], # 强调紧迫性
'P101': ["背景信息是否使用了‘平行结构’的叙事结构？", [("消息", "MESSAGE")]], # 强调紧迫性
'P102': ["该背景信息是否使用了‘损失与收益框架’的叙事框架？", [("消息", "MESSAGE")]],
'P103': ["该背景信息是否使用了‘属性框架’的叙事框架？", [("消息", "MESSAGE")]],
'P104': ["该背景信息是否使用了‘时间框架’的叙事框架？", [("消息", "MESSAGE")]], # 强调紧迫性
'P105': ["该背景信息是否使用了‘行动框架’的叙事框架？", [("消息", "MESSAGE")]], # 强调紧迫性
'P106': ["该背景信息是否使用了‘社会比较框架’的叙事框架？", [("消息", "MESSAGE")]]

        },
    # 用于格式化问题
    # Evidence: [证据文本]
    # Message: [声明文本]
    # Q: [问题]
    'G_prefix': "Evidence:EVIDENCE\n\nMessage: CLAIM\n\n",
    'Q': "Q: "
}

