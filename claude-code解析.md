# prompt

我之前深入研究过 Claude Code 的 system prompt 工程。首先说整体架构——它不是一个大字符串，而是一条**分层组装的管线**。整个 prompt 被一个叫 `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` 的标记切成两半。前半部分是身份声明、行为规范、编码原则，这些内容**对所有用户通用**；后半部分是当前工作目录、git 状态、记忆文件、MCP 服务器指令这些**会话级**的动态内容。

这么做的核心目的是**省钱**。Anthropic API 支持 prompt caching，系统提示的前缀相同就能复用之前的 KV cache 计算。静态段放前面，就可以在全量用户间共享这个缓存。它甚至专门把函数命名成 `DANGEROUS_uncachedSystemPromptSection`——谁要往缓存区加动态内容，code review 的时候这个函数名本身就是一面红旗。据它内部注释说，光是把 agent 列表从 tool schema 移到 message attachment 里，就减少了 10% 的 cache creation token 开销。

内容设计上有三个我觉得很精妙的点。

**第一个是给框架而不是给规则。**

比如它处理危险操作的方式。很多人写 prompt 会列一堆"不许删文件""不许 force push"，但 Claude Code 的做法是教模型一个**决策框架**。它的原文是：

> Carefully consider the reversibility and blast radius of actions. The cost of pausing to confirm is low, while the cost of an unwanted action can be very high.

"可逆性"和"爆炸半径"——模型拿到这两个维度之后，遇到任何新场景都可以自己推理该不该做。然后它补充了四类风险操作的例子，但用的措辞是 "Examples of the kind of risky actions that warrant user confirmation"——明确告诉模型这不是完整清单，只是帮你校准判断的参考。

最后一句收得很漂亮：

> Follow both the spirit and letter of these instructions — measure twice, cut once.

"三思而后行"。它没有试图穷举所有边界情况，而是把安全原则浓缩成一个可以泛化推理的心智模型。这种写法比规则列表鲁棒得多，因为规则总有覆盖不到的地方，但框架可以推广。

**第二个是主动对抗模型的默认偏好。**

这是我觉得最体现对 LLM 行为理解的地方。大语言模型有一些很强的默认倾向——比如倾向于多写代码、多加注释、多做抽象、多加错误处理。这些在人类工程师看来是"好习惯"，但在 AI 辅助编程的场景下会导致代码膨胀。所以 Claude Code 写了一组非常反直觉的约束：

> Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. Three similar lines of code is better than a premature abstraction.

"三行重复代码好过一个过早的抽象"——这句话直接给了模型一个具体的阈值感。不是抽象地说"不要过度设计"，而是告诉它：重复三次也没关系。

然后错误处理那条：

> Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries.

"只在系统边界做校验"——对内部代码，信任框架保证。这其实是 YAGNI 原则和防御性编程之间的一个精确切分。大多数人写 prompt 不会写到这个精度，因为你得先理解模型倾向于在哪些地方过度防御，才能写出这么有针对性的约束。

还有一个更细的例子，关于工具选择。模型天然倾向于用 Bash 做一切事情，因为 Bash 最通用。Claude Code 在 BashTool 的描述最开头就写：

> IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands. Instead, use the appropriate dedicated tool.

把"不要用我"写在工具自己的描述第一行，这种设计思路很清楚——知道模型会优先选 Bash，所以在它最可能出现的地方设置拦截。

**第三个是元认知级别的指令——教 AI 怎么给另一个 AI 写 prompt。**

Claude Code 有一个 AgentTool，就是让主模型去调度子 agent。它的 prompt 里有一节叫 "Writing the prompt"，原文是：

> Brief the agent like a smart colleague who just walked into the room — it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters.

"像给一个刚走进房间的聪明同事做 briefing"——这个比喻精确地传达了子 agent 的上下文边界。然后它给了一个非常锐利的反模式：

> **Never delegate understanding.** Don't write "based on your findings, fix the bug" or "based on the research, implement it." Those phrases push synthesis onto the agent instead of doing it yourself. Write prompts that prove you understood: include file paths, line numbers, what specifically to change.

"永远不要委托理解"——不要写"根据你的调查结果来修 bug"，因为这是把综合分析的工作推给了子 agent。你写的 prompt 应该能证明你自己已经理解了问题。这个指令解决的是一个很真实的 failure mode：主模型偷懒，把含糊的任务直接丢给子 agent，子 agent 缺少上下文后产出质量很差。

更精巧的是它在 few-shot 里设计了一个**负面场景**。它演示了用户在子 agent 还没返回结果的时候追问"那个 gate 到底接好了没有"，示范的正确回答是：

> Still waiting on the audit — that's one of the things it's checking. Should land shortly.

这个 example 的设计意图是教模型：**你不知道的事情就说不知道，不要编造中间结果**。很多人写 few-shot 只演示正确路径，但 Claude Code 专门演示了"诱惑你编造答案的场景"和对应的正确做法。这种负面场景的 few-shot 对控制幻觉非常有效。

**总结一下**，好的 system prompt 工程不只是"写提示词"。它本质上是一个系统设计问题——你需要理解模型的行为偏好才能写出有针对性的约束，需要理解 API 的缓存机制才能设计 prompt 的拓扑结构，需要理解真实的 failure mode 才能设计出有效的 few-shot。Claude Code 在这三个维度上都做得很到位。

# skill

Claude Code 的 Skills 系统最核心的设计哲学就是一句话：**Prompt 即能力**。它不需要你写任何执行逻辑代码，一个 Markdown 文件加上 YAML 配置就能封装一整套复杂工作流，这个思路我觉得非常优雅。

------

**第一层：Skill 和 Tool 的本质区别**

首先要理解 Skill 和 Tool 不是一回事。Tool 是原子操作，比如读文件、执行 bash 命令，每个 Tool 背后是一段 TypeScript 的 call 方法。而 Skill 的粒度更大，它是一套完整的工作流——比如代码审查、创建 PR。但关键是，Skill 本身不包含执行逻辑，它本质上就是一段高质量的 Prompt 加上权限声明。你告诉 AI "审查什么、按什么顺序看、输出什么格式"，剩下的事情 AI 自己用已有的 Tool 去完成。这其实是把人类专家的经验编码成了可复用的 Markdown 模板。

------

**第二层：加载架构——五个来源的统一管理**

Skill 有五个来源：内置命令、编译时打包的 Bundled Skills、磁盘上 `.claude/skills/` 目录的自定义 Skill、MCP 协议动态发现的 Skill、还有兼容旧格式的 Legacy Commands。这五个来源最终都被统一成 Command 对象，走同一套注册和执行流程。

这里面有几个设计细节我觉得值得注意。第一是 Bundled Skills 用了延迟提取，首次调用才解压，而且用闭包级 memoize 防止并发竞态。第二是磁盘 Skills 加载时用 realpath 解析符号链接来去重，避免同一个 Skill 被不同路径重复加载。第三是 MCP Skills 有一个安全守卫——禁止执行内联 shell 命令，因为远程内容不可信。这体现了一个分层信任的思想：本地 Skill 信任度高，远程的就要限制。

------

**第三层：执行模型——Inline 和 Fork 双模式**

Skill 执行有两条路径。默认是 Inline 模式，把 Prompt 注入到当前对话流里，AI 在主上下文中继续执行。另一个是 Fork 模式，会启一个独立的子 Agent，有自己的 token 预算，执行完把结果提取回来，然后释放子 Agent 的所有消息。

这个设计很实用。像简单的格式化、小改动用 Inline 就行；但如果是长时间运行的代码审查，用 Fork 可以避免污染主对话的上下文窗口。而且 Fork 执行完后会主动清理状态，这对上下文管理很重要。

------

**第四层：资源管控——预算截断和条件激活**

Skills 系统在资源管理上做得很细。首先是 Prompt 预算，所有 Skill 的描述信息注入 System Prompt 时只占上下文窗口的 1%，单条描述上限 250 字符。如果超了，Bundled Skills 保持不动，只压缩用户自定义的。有三级降级：完整描述 → 均分剩余预算 → 只保留名字。

然后是条件激活机制。带 paths 字段的 Skill 平时是不可见的，存在一个 conditionalSkills Map 里。只有当 AI 实际操作了匹配路径的文件，比如编辑了 `*.test.ts`，测试相关的 Skill 才会被激活。这避免了不相关的 Skill 浪费上下文空间。

还有使用频率排名，用指数衰减算法，7 天半衰期，最近常用的排前面，但老的高频 Skill 有 0.1 的保底分不会完全消失。

------

**第五层：安全设计——正向白名单**

权限检查是四层：Deny → 官方市场放行 → Allow → Safe Properties 白名单。最后一层用的是正向安全设计，只有 28 个明确安全的属性可以免确认，未来新增的任何属性默认都要用户授权。这比反向黑名单更安全，因为黑名单容易遗漏，白名单的默认行为是拒绝。

------

**收尾总结**

总结一下，Claude Code 的 Skills 系统给我最大的启发有三点。第一，把"经验"而不是"逻辑"作为封装单元，这在 Agent 架构里是一个很有意思的范式。第二，在资源管控上非常精细，预算截断、条件激活、使用频率排名，都是为了在有限的上下文窗口里最大化信息密度。第三，安全设计用了分层信任加正向白名单，这在可扩展系统里是一个很好的实践。如果让我类比的话，Skill 系统有点像是给 AI Agent 做了一套"SOP 管理平台"，每个 Skill 就是一份标准操作流程。

# 上下文

## 记忆

下面是面试口语稿：

------

**开场（一句话结论）**

Claude Code 的项目记忆系统最让我印象深刻的一点是：它完全基于文件，没有数据库、没有向量存储，就是 Markdown 文件加目录结构。这个设计非常克制，但通过一套精心设计的分类法、智能召回和漂移防御机制，做到了实用且可靠的跨对话记忆。

------

**第一层：存储架构——纯文件、零依赖**

整个记忆系统存在 `~/.claude/projects/<项目路径>/memory/` 下面，入口是一个 MEMORY.md 索引文件，每次对话完整加载。索引有双重上限：200 行和 25KB，超过就截断。为什么要双重上限？因为 p97 的索引 200 行就够了，但有时单行特别长，字节上限是用来捕捉这种异常的。

有一个安全细节值得注意：记忆路径的自定义配置故意排除了 projectSettings 层级，防止恶意仓库把记忆路径指向 `~/.ssh` 这种敏感目录。同一个 Git 仓库的所有 worktree 通过找到 canonical git root 来共享同一个记忆目录，这保证了分支切换不会丢失记忆。

------

**第二层：四类型分类法——封闭类型系统**

记忆不是随便存的，被约束为四种类型：user（用户角色偏好）、feedback（用户对 AI 行为的纠正和确认）、project（项目上下文）、reference（外部系统指针）。这是一个封闭类型系统，每种类型都有明确的存储条件和使用方式。

核心约束是：**只存无法从当前项目状态推导的信息**。代码架构、文件路径、git 历史都能实时获取，不需要记忆。这避免了记忆和现实不一致的问题。

这里面 feedback 类型的设计特别有意思。它强调"双通道捕获"——不仅在用户说"不要这样做"时保存，也要在用户说"对，就是这样"时保存。只记失败不记成功的话，AI 会变得过度保守，行为会随时间漂移。这跟强化学习的思路很像，你需要正反馈和负反馈都有。

每条记忆都有一个 frontmatter，其中 description 字段不是给人看的摘要，而是给 AI 召回系统做相关性判断的搜索关键词，这一点设计得很巧妙。

------

**第三层：智能召回——Sonnet 侧查询 + 去噪**

不是所有记忆都适合每次对话。系统用一个轻量级的 Sonnet 侧查询来做筛选，从所有记忆文件的 frontmatter 清单里选出最多 5 条最相关的。

这里有两个去噪机制很实用。第一个是近期工具去噪：如果 AI 正在用某个工具，就不需要召回那个工具的使用文档了，因为工作上下文已经在对话里了。但它做了一个精细的区分——工具的使用文档跳过，但关于这个工具的**警告、陷阱、已知问题**仍然要召回，因为这正是使用时最需要的信息。第二个是已展示去重：之前轮次已经展示过的记忆文件不会再选，让 5 个召回槽位花在新的候选上。

------

**第四层：记忆漂移防御——不盲目信任记忆**

这是我觉得最成熟的设计。系统在 Prompt 里专门有一段叫"Before recommending from memory"，要求 AI 在使用记忆之前做验证：记忆里提到的文件路径要先检查是否存在，提到的函数或 flag 要先 grep 一下。因为记忆写入时代码可能已经被重命名或删除了。

还有一个细节，关于"忽略记忆"的严格语义。用户说"忽略关于 X 的记忆"，AI 应该表现得好像 MEMORY.md 是空的，不能引用、不能比较、不能提及。这解决了一个常见反模式——AI 说"虽然记忆里说了 Y，但我们不用它"，这不是忽略，这是承认然后覆盖。据文章说，"Before recommending from memory"这个标题措辞还经过了 A/B 测试，行动导向的写法比抽象描述效果好很多。

------

**第五层：KAIROS 模式和压缩联动**

对于长期运行的 assistant 会话，标准的实时索引维护太频繁了，所以有一个 KAIROS 模式：AI 只往按日期组织的日志文件追加内容，不做实时重组。然后由一个独立的夜间 `/dream` 技能来蒸馏日志，生成主题文件和更新 MEMORY.md 索引。这相当于把"记录"和"整理"解耦了。

记忆系统和上下文压缩也有深度联动。开了 Session Memory 的 feature flag 之后，压缩时优先用 Session Memory 做摘要，不需要调 API 生成摘要，更快更便宜。

------

**收尾总结**

总结一下，这套记忆系统给我三个核心启发。

第一，**存储选择的克制**。纯文件方案看起来原始，但它零依赖、易调试、版本可控。在 Agent 系统里，过度工程化存储反而增加故障面。这跟我做 Paper Pilot 项目时的体会一致——我们一开始用 Elasticsearch 做检索，后来整合到 Milvus 2.5 的原生 BM25，减少外部依赖反而更稳定。

第二，**分类法比自由存储更重要**。四种类型加上"只存不可推导信息"的约束，让记忆系统不会退化成垃圾场。在我的金融多 Agent 系统里做结论存储也有类似的思路，需要区分结论、证据、局限性，而不是什么都往一个地方扔。

第三，**不盲目信任记忆**。记忆写入时的世界状态和召回时的世界状态可能不一样，所以必须有验证机制。这个思路在 RAG 系统里也完全适用——检索到的文档不一定还是最新的，使用前需要 grounding check。

------

时间紧的话，讲开场 + 第二层（四类型分类法）+ 第四层（漂移防御）+ 收尾就够了，大概两三分钟。面试官追问再展开召回机制或 KAIROS 模式。

## 压缩

Claude Code 的上下文压缩系统，核心设计思路是**三层递进式策略**——从局部清理到本地压缩再到 API 摘要，每层成本递增但压缩力度也递增。这个分层设计在工程上非常实用，因为你不希望每次上下文快满了都去调一次摘要 API，那太贵也太慢了。

------

**第一层：MicroCompact——局部压缩，零成本**

最轻量的一层叫 MicroCompact，它不压缩整个对话，只做一件事：把旧的工具输出内容清掉。它维护了一个白名单，包含 Read、Bash、Grep、WebFetch 这些常见工具，超过时间窗口的工具输出会被替换成一句 `[Old tool result content cleared]`。原始内容还在 JSONL transcript 里，只是不再发给 API。

这层设计很聪明，因为在 Agent 场景里，工具输出往往是上下文里最占空间的部分——你读一个文件可能几千 token，一次 bash 输出也可能很长。但越老的工具输出对当前任务的参考价值越低，所以用时间衰减来决定清除优先级。图片和 PDF 超过 2000 token 也会被清掉。关键是这一层完全不需要 API 调用，零额外成本。

------

**第二层：Session Memory Compact——用已有记忆替代 API 摘要**

第二层叫 Session Memory Compact，这是一个很有意思的设计。它的前提是系统已经在后台维护了 Session Memory，也就是从对话中持续提取的结构化记忆。所以压缩的时候，它不需要额外调 AI 模型去生成摘要，直接用已有的 Session Memory 作为对话摘要就行了。

压缩后保留多少消息，有一个双下限加上限的算法：至少保留 10K token 和 5 条文本消息，保证有足够的上下文深度和对话连续性；最多保留 40K token，避免压完还是太大又触发下一次压缩。

这层有一个关键的正确性保证：工具对完整性保护。因为 API 要求每个 tool_result 必须有对应的 tool_use，如果压缩的切割点恰好把 tool_use 切掉了，保留了孤立的 tool_result，API 就会报错。所以它有一个 `adjustIndexToPreserveAPIInvariants` 函数，会向前扫描确保切割点不破坏工具对的完整性。而且因为流式传输会把一个 assistant 消息拆成多条记录，这增加了边界情况的复杂度。

------

**第三层：传统 API 摘要——最后的兜底**

如果前两层都不可用，比如 Session Memory 的 feature flag 没开，或者用户用了自定义指令（比如 `/compact 聚焦在认证模块`），就走传统路径：调 AI 模型对整段对话生成摘要。

发给摘要模型之前会做预处理，图片替换成 `[image]` 文本标记，避免摘要调用本身也触发 prompt too long。压缩之后还有一个重新注入机制，有 50K token 的预算用来恢复关键上下文：最近读过的文件（最多 5 个，每个 5K token）、已激活的 Skill 指令（总共 25K token）、CLAUDE.md 的项目记忆、还有 MCP 工具发现结果。这个设计的意思是，压缩不是无差别丢弃，而是丢掉过程细节、保留关键状态。

------

**边界机制：CompactBoundary**

每次压缩后会在消息流里插入一条 CompactBoundary 标记，记录压缩类型、压缩前 token 数、保留了哪些消息的 UUID。后续所有操作都只处理最后一条 boundary 之后的消息。这个标记还带了 preservedSegment 注解，在会话恢复时帮助加载器重建消息链，避免重复压缩已保留的消息。