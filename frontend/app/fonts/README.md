# Inter 本地字体来源

## 固定版本与授权

- 上游项目：[rsms/inter](https://github.com/rsms/inter)
- 固定发布版本：[Inter 4.1（v4.1）](https://github.com/rsms/inter/releases/tag/v4.1)
- 字体来源：[InterVariable.woff2](https://raw.githubusercontent.com/rsms/inter/v4.1/docs/font-files/InterVariable.woff2)
- 许可证来源：[LICENSE.txt](https://raw.githubusercontent.com/rsms/inter/v4.1/LICENSE.txt)
- 授权：SIL Open Font License 1.1；版权归 The Inter Project Authors。
- 字体及许可证均保留上游原始字节，随项目分发时保留 LICENSE.txt。
- 上游固定标签的字体文件内部版本为 `Version 4.001;git-9221beed3`。

## 文件 SHA256（对下载原始字节的本地计算）

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| InterVariable.woff2 | 352240 | `693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3` |
| LICENSE.txt | 4380 | `262481e844521b326f5ecd053e59b98c8b2da78c8ee1bdbb6e8174305e54935a` |

## 格式、字重和字符覆盖

用项目已有 Next.js 14.2.35 随附的 fontkit 解析，并用系统 fc-scan 交叉核对：

- 实际文件头为 wOF2，格式为 WOFF2。
- 字体族 Inter Variable，常规样式，PostScript 名称 InterVariable。
- wght 轴：100–900，默认 400；opsz 轴：14–32，默认 14。
- 字形数 2937，字符映射含 2852 个 Unicode 码点。
- 覆盖全部 95 个 Basic Latin 可打印字符。
- 检查了现有 8 种语言的 144 个翻译目录文件，其中 77 种拉丁及组合字符全部有字形。
- 另检查了 53 种扩展拉丁样例，包含 Āā、Ąą、Čč、Đđ、Ğğ、İı、Łł、Œœ、Šš、Žž、Ǎǎ、ẞ，全部有字形。

| 检查的 Unicode 范围 | 已覆盖 / 范围码点数 | 未覆盖码点 |
| --- | --- | --- |
| U+0020–U+007E | 95 / 95 | 无 |
| U+00A0–U+00FF | 95 / 96 | U+00AD |
| U+0100–U+017F | 127 / 128 | U+0149 |
| U+0180–U+024F | 207 / 208 | U+01C4 |
| U+1E00–U+1EFF | 255 / 256 | U+1E9C |

原来的 latin、latin-ext 是 Google Fonts 的子集名称，并不保证这些 Unicode 区段的每个码点都有字形。
本文件包含满足当前拉丁文案及扩展拉丁使用需要的字形；上表如实保留例外，超出字体覆盖的字符继续使用现有系统字体回退链。
中日韩文字也继续通过原有系统字体回退处理。

## 项目接入

根布局使用 next/font/local 读取本目录文件，保持 --font-inter、100–900 字重、normal 样式及 display: swap。
Tailwind 的 font-sans 和既有回退字体列表保持原样。
字体文件纳入源码，使后续构建从本地读取；验证时继续禁止外部网络请求。
