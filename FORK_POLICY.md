# 《999》二开仓策略（FORK_POLICY）

> 本文件为《999》项目对 asuna778548/N.E.K.O 的治理声明，覆盖 fork 的一切同步决策。
> 上游：Project-N-E-K-O/N.E.K.O；基线锚定提交：`3fd9c29476b52ce351bde555d80e72cf611012fd`（companion ADR `puosui/docs/reviews/companion-runtime-final-adr-2026-08-03.md` 明文锚定）。

## 1. 定位

本仓库是《999》的 **N.E.K.O 二开仓**（companion 语音双链客户端：MicBroker 唯一麦克风 / UniqueAudioSink 唯一播放时钟 / Overlay 宿主）。它不是上游的跟踪镜像。

## 2. 同步规则

1. **独立二开，不与上游合并**：日常禁 `git merge/rebase/cherry-pick` 上游任何分支。
2. **禁用 GitHub "Sync fork" 按钮**：它会快进 main 到上游、制造分叉（2026-09-07 曾发生：origin/main 被同步到上游 #3002，已以二开 main e9ff5d01 覆盖回滚；镜像点留痕于本地 tag `upstream-mirror-synced-20260907`）。
3. **上游演进只在主创显式裁决时评估**：评估动作=新开裁决卡，产出重放/迁移方案后再动 main。
4. 二开分支（wave4/input、wave4/output、backup-main-20260905）内容均已并入 main（merge commits 0aaab0b / 8c86da0）；保留作过程存档。

## 3. 分支语义

| 分支 | 含义 |
|---|---|
| `main` | 二开主线（基线 3fd9c29 + 语音双链等增量） |
| `v1/main` | B01 合流时点快照（29e0ca4，已含于 main） |
| `backup/wave10-main-before-merge-20260901` | 基线 3fd9c29 锚 |
| `backup-main-20260905`（远端） | 二开 main 的远端备份（= e9ff5d01） |
