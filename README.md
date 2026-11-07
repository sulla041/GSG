# GSG GitHub 仓库

`hub/` 是本项目的 GitHub 仓库目录，用于保存和同步仓库内容，不是普通的临时工作目录。

仓库操作约定登录名为 `sulla041`。任何登录、拉取、提交、推送或其他远程操作，都必须在获得明确授权后执行。

## 当前数据文档状态（2026-10-29）

根目录与本目录 `docs/` 的工程结构、试验记录、算法描述、具体设计和行为准则保持镜像同步。CircuitNet 当前有 `10,305` 张 `G_topo` 与 `10,305` 张 `G_delay_input`；后者已显式保存 cell master、drive、slew/load 和 direct arc 输入，但因缺少可核验 Liberty/PVT/SDC/activity/physical 条件，状态为 `condition_incomplete_external`，不进入对应任务的正式训练统计。

截至本次同步，Hub 本地 `main` 与已获取的 `origin/main` 指向同一提交；上述文档改动仅在本地工作区，尚未执行 commit、pull、merge 或 push。
