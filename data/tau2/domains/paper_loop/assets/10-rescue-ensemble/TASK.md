# 10-rescue-ensemble：森林火灾救援集成优化

本案例属于动态救援仿真与资源优化。需要以论文《Ensemble framework by using nature inspired algorithms for the early-stage forest fire rescue — A case study of dynamic optimization problems》为主要证据，持续完善已有 NumPy 核心重建。

## 输入材料

论文文件为本案例目录中的 `paper.pdf`，登记的 SHA-256 为 `9a3bd1a03fd6fdc71b96f18f5ccb0b4ddffe2d3cbdb315be381a1b3aa4690921`。第一轮从本案例目录中的 `starter/` 候选开始；它只是可修改的起点，不是标准答案。

## 本案例重点

重点核对动态火场状态、关键火点选择、FPSO 飞机分配和反馈闭环。每个执行步骤都应能关联论文中的页码、公式、算法或图，并能对应到候选中的具体代码和中间结果。

## 完成标准

- 候选明确区分论文事实、工程推断和为了可运行而采用的替代假设。
- 执行图、说明、NumPy 代码和实际输出彼此一致，没有仅存在于图中或仅存在于代码中的关键步骤。
- 实现只使用 Python 标准库与 NumPy，可在 CPU 和无网络环境中运行，并在固定种子下重复得到一致结果。
- 原始输出足以让外部检查重新计算目标值和约束；候选自报的通过状态不能代替独立验证。
- 对论文未公开的数据、地图、权重、训练产物或商业工具结果，必须说明替代方式及其对结论的影响。

每轮开发和审查都应提交自然语言 Markdown 记录，不得在面向人的报告中倾倒 JSON、YAML、XML 或内部控制状态。独立审查者每轮使用全新上下文，且只能查看论文、本页要求、当前冻结候选和本轮自动检查结果。
