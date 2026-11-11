# 简历描述（route-audit）

> 用途：直接粘贴进简历「项目经历」栏。按你投递的岗位选一版。
> 所有数字均可在仓库 README 复现（`python -m routeaudit report --sweep`）。

---

## 一句话版（简历标题行）

**模型路由成本审计沙盒**｜Python · 离线评估 · 成本归因｜个人项目

---

## 标准版（推荐，4–5 行）

**模型路由成本审计沙盒**（Python，个人项目）

- 针对 LLM 级联路由"只报省了多少、不报省得对不对"的盲区，构建离线审计框架：
  用标注集构造**反事实矩阵**，让任意阈值的成本-准确率曲线可零 token 离线重算。
- 在 200 任务 × 3 策略 × 9 阈值上实测：级联比全高档模型**省 76.5% 成本**，
  但其中 **62.8% 的支出属于路由决策错误**（实际 $0.6911 vs 精度匹配 oracle $0.2572）。
- 发现三类反直觉结论：过度升级仅占 token 5.5% 却占成本 24.4%（价格比 16.67×）；
  级联比全高档**多用 6.1% token**（省的是钱不是 token）；级联准确率**非随阈值单调**
  （θ=0.8 比 θ=0.7 更低却贵 34%）。
- 工程实现：537 次调用覆盖 4000 次调用的实验空间（反事实重放 + 缓存）、
  预算守卫、ECE/Brier 校准诊断、regret 分析、Markdown 报告与三张可视化；
  373 个标准库 unittest 测试，全离线可复现。

---

## 精简版（3 行，适合版面紧张）

**模型路由成本审计沙盒**（Python）

- 用反事实矩阵做 LLM 级联路由的离线成本归因，量化"路由错误花了多少冤枉钱"：
  实测级联省 76.5% 成本，但 62.8% 是路由 regret。
- 揭示"过度升级占 token 5.5% 却占成本 24.4%"的非对称性，及阈值到准确率的非单调关系。
- 373 个标准库测试全绿，537 次调用覆盖 4000 次调用的实验空间。

---

## 英文版（投外企用）

**Model-Routing Cost Audit Sandbox** (Python, personal project)

- Built an offline audit framework that prices *routing mistakes* rather than
  reporting routing savings. A counterfactual matrix (cheap-only vs premium-only
  graded over a labelled set) makes any cascade threshold's cost/accuracy curve
  recomputable offline with zero tokens.
- On 200 tasks the cascade is **76.5% cheaper** than premium-only, yet **62.8%
  of its spend is routing regret** ($0.6911 actual vs $0.2572 for an
  accuracy-matched oracle; 31 escalations vs 13).
- Found three counter-intuitive results: over-escalation is 5.5% of tokens but
  24.4% of cost (16.67× price ratio); the cascade uses **6.1% more tokens** than
  premium-only; and cascade accuracy is **non-monotone** in the threshold.
- 537 provider calls cover a 4000-call experiment space; 373 stdlib-only tests.

---

## 面试可展开的 5 个技术点

1. **为什么现有指标不够**：CostGuard / AgentFuse 报"省了多少"，但没有分母——
   相对"全用最贵模型"这个没人会选的基线。真正该问的是"这个准确率理论上最低要
   多少钱"。离线审计有 gold label，所以能算这个下界。
2. **反事实矩阵**：策略 A/B 各跑一遍并逐题评分，得到每题的
   `(cheap_verdict, premium_verdict)`；之后每个阈值只是纯重放。这是把
   "9 阈值扫描"从 4000 次调用压到 537 次的核心，也让报告可离线复算。
3. **token 占比会骗人**：过度升级占 token 5.5%、占成本 24.4%。因为高档模型比廉价
   贵 16.67×。任何"用 token 衡量浪费"的说法在这个配置下都是错的。
4. **准确率非单调**：8% 的任务上高档模型答错而廉价答对，升级会把对的改错。所以
   θ=0.8 的准确率比 θ=0.7 更低、成本更高。"调高阈值换准确率"这个直觉是错的。
5. **测量工具本身要被测量**：审计过程中我们自己的代码被查出 5 个缺陷
   （格式门控的死代码、阈值键冲突等），全部修复并加了回归测试。

---

## 可能被追问的问题 & 回答要点

- **这些数字是真的调 API 跑的吗？** 不是，默认是**确定性校准模拟器**。
  诚实回答：选择模拟器是为了可复现（同 seed 同结果）、零成本、可做消融实验。
  方法论不变，换真实 API 只需 `--provider api`，具体数字会变。
  README 把校准参数全部公开，供质疑。
- **为什么不直接用 LiteLLM？** 审计需要完全掌控每一步的 token 记账与门控时机，
  网关抽象反而挡路。真实 API 适配层保留了同样的接口。
- **结论能推广到生产吗？** 不能直接推广——它依赖价格比与任务难度分布。所以代码里
  有 `--price-scale` 和可替换的 `CATALOGUE`。可迁移的是**方法论**（归因结构、
  非对称性、非单调性），不是小数点。
- **和路由框架有什么区别？** 那些是"执行层"（帮你路由），这是"审计层"
  （告诉你路由决策错在哪）。两者互补。
