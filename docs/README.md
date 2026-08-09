# 项目文档

- `dataset-taxonomy.md`：宿主作物、四大标签属性和 203 个细粒度类别的层级规范；
- `project-plan.md`：20 天里程碑、硬件判断和模型路线；
- `development-guide.md`：数据、训练、评估和应用开发规范；
- `team-responsibilities.md`：数据、算法、应用三位 AI 工程师的职责和交接标准；
- `workflow.md`：用户逐阶段调用工程师、阶段停点、返工和汇报规则；
- `acceptance-checklist.md`：由总负责人维护的阶段状态与复验清单；
- `prompts/`：三位 AI 工程师的固定启动提示词；
- `worklogs/`：三位工程师的实施日志和总负责人的验收日志；

数据集事实、论文勘误和翻译材料位于 `research/`，结构化类别元数据位于 `metadata/`。

执行项目时先读 `workflow.md`，再按 `acceptance-checklist.md` 确认当前阶段。新会话先使用 `prompts/` 恢复角色上下文，并读取 `worklogs/` 中的历史记录。工程师不能根据计划表自行宣告阶段完成，只有总负责人复验后可以修改验收状态。
