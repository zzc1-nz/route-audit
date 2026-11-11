# 上传到 GitHub 操作指南

> 目标仓库：<https://github.com/zzc1-nz/route-audit>（public）

## 为什么需要你手动执行

这台机器**所有 HTTPS 都不通**（不只是 GitHub——`example.com`、`pypi.org` 都失败，
报 `SEC_E_NO_CREDENTIALS`）。所以 `git push` 只能在你本机完成。

本地仓库已经准备好：2 个 commit，工作区干净，35 个文件，只差一条 push。

---

## 方案 A：本机已装 Git（推荐，最快）

### 1. 在 GitHub 网页上建空仓库

打开 <https://github.com/new>：

| 字段 | 填什么 |
|---|---|
| Repository name | `route-audit` |
| Description | `Offline audit sandbox for model-routing cost attribution` |
| Visibility | **Public** |
| Initialize with README / .gitignore / license | **全部留空不勾** |

> 关键：**不要**勾选任何初始化选项。否则远程会有一个初始 commit，
> 你 push 时会因历史不一致被拒。

### 2. 在本机打开 PowerShell，进入项目目录

如果项目还在原来位置：

```powershell
cd "C:\Users\24657\Documents\deepseek-harness\default-workspace\route-audit"
```

### 3. 推送

```powershell
git push -u origin main
```

首次推送会弹窗要求登录 GitHub。推荐用 **Personal Access Token**（不是密码）：

1. 打开 <https://github.com/settings/tokens> → **Generate new token (classic)**
2. Note 填 `route-audit`，Expiration 选 90 天，勾选 **`repo`** 权限
3. 生成后**立刻复制** token（页面关闭就看不到了）
4. 回到 PowerShell，Username 填 `zzc1-nz`，Password 粘贴 token

推送成功后打开 <https://github.com/zzc1-nz/route-audit> 确认。

---

## 方案 B：用 git bundle 传输（项目不在本机时）

如果你要从**另一台机器**上传，先拷走 bundle 文件：

```
route-audit\route-audit.bundle   （约 400 KB，含完整 2 个 commit）
```

在目标机器上：

```bash
git clone route-audit.bundle route-audit
cd route-audit
git remote set-url origin https://github.com/zzc1-nz/route-audit.git
git push -u origin main
```

> bundle 是完整历史，克隆出来和原仓库一模一样。
> 传输前可先校验：`git bundle verify route-audit.bundle`
> 应输出 `The bundle records a complete history.`

---

## 方案 C：完全不用 Git（最省事，但丢历史）

1. 在 GitHub 上新建仓库 `route-audit`（同上，不勾初始化）
2. 点 **uploading an existing file**
3. 把 `route-audit` 目录下**除 `runs/`、`out/`、`.git/` 之外**的所有文件拖进去

> 不推荐：会丢失两个有意义的 commit message，而 commit 历史本身是
> 代码质量的信号（招聘方会看）。

---

## 推送后建议做的 4 件事

### 1. 加 Topics（提升可搜索性）

仓库页右上 **⚙️ Settings → Topics**，添加：

```
llm  llm-routing  cost-optimization  model-cascade  evaluation
llmops  cost-attribution  python  audit
```

### 2. 确认 README 在首页正常渲染

打开仓库首页，检查三张图是否显示。若图片 404，通常是
`docs/figures/` 没推上去，执行 `git ls-files docs/figures` 核对。

### 3. 把仓库加到简历的链接

简历里写：

```
github.com/zzc1-nz/route-audit
```

### 4. 如果你想让它出现在 GitHub 主页

在 <https://github.com/zzc1-nz> 主页，仓库默认按最近更新排序，
确保这个仓库是最近 push 的即可置顶。也可以点仓库右上 **Pin** 固定。

---

## 验证清单（push 完成后自查）

在仓库页面逐一确认：

- [ ] 首页 README 正常显示，含 3 张图
- [ ] 文件树里有 `routeaudit/`（14 个 .py）、`tests/`（10 个文件）
- [ ] **没有** `runs/`、`out/`、`__pycache__`（被 .gitignore 排除）
- [ ] 顶部显示 2 次 commit
- [ ] LICENSE 显示 MIT
- [ ] Topics 已添加

然后别人 clone 下来应该能直接跑：

```bash
git clone https://github.com/zzc1-nz/route-audit
cd route-audit
python -m unittest discover -s tests        # 373 passed
python -m routeaudit report --sweep --out-dir out/
```

---

## 常见问题

**Q: `git push` 报 `rejected (fetch first)`？**
说明远程仓库被初始化过（勾了 README）。解决：

```powershell
git pull --rebase origin main
git push -u origin main
```

**Q: 报 `remote: Support for password authentication was removed`？**
GitHub 不再接受账号密码，必须用 Personal Access Token（见方案 A 第 3 步）。

**Q: 图片在 GitHub 上显示不出来？**
检查 `git ls-files docs/figures` 有没有输出。没有的话：

```powershell
git add -f docs/figures
git commit -m "docs: add figures"
git push
```

**Q: 想改仓库名或改 public/private？**
Settings → General → 最上方 Rename / 最下方 Danger Zone → Change visibility。

**Q: 推送后发现有敏感信息？**
本仓库没有密钥或个人信息（全部是模拟数据），可以放心公开。
`runs/audit.json`（4 MB 审计产物）已被 .gitignore 排除。
