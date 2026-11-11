# 上传到 GitHub 操作指南（PAT Classic 路线）

> 目标仓库：<https://github.com/zzc1-nz/route-audit>（public）
>
> 本地仓库已就绪：3 个 commit、37 个文件、工作区干净、远程 origin 已配置。
> 你只需要：建 token → 建仓库 → push。

---

## ⚠️ 两条机器级注意事项（先读，否则 push 必失败）

这两条是这台机器实测发现的真实障碍，与 token 无关：

### 1. git 全局代理指向一个没在运行的端口

这台机器的 git 全局配置里有：

```
http.proxy  = http://127.0.0.1:7897
https.proxy = http://127.0.0.1:7897
```

如果你**没有开代理软件**（Clash/v2ray 等），push 会直接报
`Failed to connect to 127.0.0.1 port 7897`。二选一：

```powershell
# 选择一：先把你的代理软件打开（端口确认是 7897），什么都不用改

# 选择二：不用代理，直连（如果你网络可以直连 GitHub）
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### 2. 如果报 `schannel: SEC_E_NO_CREDENTIALS`

沙箱/受限环境下 Windows TLS 后端可能报这个错。给这一条命令临时换 OpenSSL 后端：

```powershell
git -c http.sslBackend=openssl push -u origin main
```

（已在本机实测：OpenSSL 后端可以正常连通 GitHub。日常终端通常不会遇到，
遇到再用这条。）

---

## 第 1 步：生成 PAT (Classic)

1. 打开 <https://github.com/settings/tokens>
2. **Generate new token → Generate new token (classic)**
3. 填写：
   - **Note**：`route-audit push`
   - **Expiration**：90 天（或你习惯的周期）
   - **Select scopes**：勾选 **`repo`** 这一项就够了
     （classic 的 `repo` scope 同时覆盖 public 和 private 仓库的读写）
4. 点页面底部 **Generate token**
5. **立刻复制**显示出来的 `ghp_...` 字符串——离开这个页面就再也看不到了

> 安全要求：token 只出现在你自己的终端里。**不要**粘贴到任何聊天窗口、
> 截图、issue 或代码里。本对话之前出现过一次 token 泄露，
> GitHub 的 secret scanning 会在几分钟内自动撤销泄露的 token——
> 这是它失效的原因，也是这个机制在保护你。

---

## 第 2 步：创建空仓库

打开 <https://github.com/new>：

| 字段 | 填什么 |
|---|---|
| Repository name | `route-audit` |
| Description | `Offline audit sandbox for model-routing cost attribution` |
| Visibility | **Public** |
| Initialize this repository with | **三个选项全部不勾**（README / .gitignore / license 都不要） |

> **为什么必须留空**：勾了任何一项，GitHub 会生成一个初始 commit，
> 和本地历史对不上，push 会被拒绝（可以补救，但多一步）。

---

## 第 3 步：推送（三选一）

打开 PowerShell，进入项目目录：

```powershell
cd "C:\Users\24657\Documents\deepseek-harness\default-workspace\route-audit"
```

### 方式一：交互式输入（最干净，推荐）

```powershell
git push -u origin main
```

- 弹出 **Git Credential Manager** 浏览器窗口时：直接关掉它，
  终端会退回命令行提问模式
- `Username for 'https://github.com'`：输入 `zzc1-nz`
- `Password for 'https://zzc1-nz@github.com'`：**粘贴 PAT**（粘贴后屏幕不显示，正常）
- 成功后 Git Credential Manager 会把凭据存进 Windows 凭据管理器，
  以后 push 不再需要 token

### 方式二：一次性内嵌 token（快，但有痕迹）

```powershell
git -c credential.helper= push https://zzc1-nz:<你的PAT>@github.com/zzc1-nz/route-audit.git main:main
git branch --set-upstream-to=origin/main main 2>$null
```

> ⚠️ token 会留在 PowerShell 历史文件里。用完执行一次清理：
> ```powershell
> # 查看并删掉包含 ghp_ 的历史行
> Get-Content (Get-PSReadLineOption).HistorySavePath | Where-Object { $_ -notmatch "ghp_" } | Set-Content (Get-PSReadLineOption).HistorySavePath
> ```

### 方式三：让 Credential Manager 全程接管

什么都不做，直接 `git push -u origin main`，在弹出的浏览器窗口里登录 GitHub
账号（不用 token）。GCM 会用 OAuth 存凭据。如果你不喜欢管理 token，
这条路最省心——但既然你要用 PAT，方式一最对口。

---

## 第 4 步：验证

```powershell
git ls-remote https://github.com/zzc1-nz/route-audit.git
# 应输出一行 d8b0... refs/heads/main 之类的 hash

git log --oneline origin/main   # 应显示 3 个 commit
```

浏览器打开 <https://github.com/zzc1-nz/route-audit>，确认：

- [ ] README 正常渲染，三张图（docs/figures/*.png）能显示
- [ ] 文件树有 `routeaudit/` 和 `tests/`，**没有** `runs/`、`out/`
- [ ] 顶部显示 3 commits，LICENSE 显示 MIT

---

## 第 5 步：收尾（可选但建议）

### 加 Topics

仓库页 About 旁 **⚙️** → Topics，添加：

```
llm  llm-routing  cost-optimization  model-cascade  evaluation  llmops  python  audit
```

### 用完撤销 token

如果这个 token 不会再用于其他仓库，到
<https://github.com/settings/tokens> 点 **Delete** 撤销。
凭据管理器里已存的登录不受影响（那是 OAuth 凭据，不是这个 token）。

### 以后日常更新

```powershell
git add -A
git commit -m "..."
git push            # 凭据管理器已存，无需再输 token
```

---

## 排错速查

| 报错 | 原因 | 处理 |
|---|---|---|
| `Failed to connect to 127.0.0.1 port 7897` | git 全局代理指向没运行的代理 | 见页首注意事项 1 |
| `schannel: SEC_E_NO_CREDENTIALS` | Windows TLS 后端在受限环境 | 加 `-c http.sslBackend=openssl` |
| `remote: Repository not found` | 仓库没建，或 token 没勾 `repo` scope | 建仓库 / 重新生成 token |
| `Authentication failed` | token 过期/撤销/复制不全 | 重新生成，完整复制 `ghp_` 开头全串 |
| `rejected (fetch first)` | 远程被初始化过（勾了 README） | `git pull --rebase origin main` 再 push |
| `remote: Support for password authentication was removed` | 把账号密码当 token 用了 | Password 处必须粘贴 `ghp_` 开头的 PAT |
| 图片 404 | `docs/figures` 没推上去 | `git ls-files docs/figures` 核对，缺则 `git add -f docs/figures` 后重新 commit+push |

---

## 附：不经过 Git 的替代方案（不推荐，仅救急）

GitHub 网页 **uploading an existing file** 可以拖文件上传，但会丢掉 3 个有意义的
commit（commit 历史本身是招聘方会看的信号），且无法上传超过 100 个文件。
仅在完全无法使用 git 时使用。

完整的仓库快照备份：项目根目录下的 `route-audit.bundle`（约 400 KB，
含全部历史），在任何机器上 `git clone route-audit.bundle route-audit` 即可还原。
