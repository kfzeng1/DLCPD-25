# 项目文档

- `dataset-taxonomy.md`：宿主作物、四大标签属性和 203 个细粒度类别的层级规范；
- `project-plan.md`：剩余阶段、硬件判断和模型路线；
- `development-guide.md`：数据、训练、评估和应用开发规范；
- `team-responsibilities.md`：数据、算法、应用三位 AI 工程师的职责和交接标准；
- `workflow.md`：用户逐阶段调用工程师、阶段停点、返工和汇报规则；
- `acceptance-checklist.md`：由总负责人维护的阶段状态与复验清单；
- `prompts/`：三位 AI 工程师的固定启动提示词；
- `workplans/`：数据 T0、算法 J1-J4、应用 J5 的执行工作单；
- `worklogs/`：三位工程师的实施日志和总负责人的验收日志；

数据集事实、论文勘误和翻译材料位于 `research/`，结构化类别元数据位于 `metadata/`。

历史分类基线 D0-F0 与 IP102 T0 已完成。当前采用双数据集交替联合训练，按 `J1-J5、F1` 推进；开始工作时先读 `project-plan.md`、`workflow.md`、自己的提示词和 `workplans/` 工作单。`acceptance-checklist.md` 是唯一状态表。
