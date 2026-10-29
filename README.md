# 🏭 AI 工业设备预测性维护平台 (AI Predictive Maintenance Platform)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-AI-orange.svg)
![Zhipu AI](https://img.shields.io/badge/LLM-GLM_4.7_Flash-green.svg)

> **基于孤立森林 (Isolation Forest) 与大语言模型 (LLM) 的智能设备诊断系统。**
> 本项目针对工业场景中设备非计划停机带来的高昂成本，通过分析多维传感器时序数据，实现设备健康状态的实时监测与智能预警。并结合大模型，实现从“发现问题”到“解决问题”的闭环。

---

## 🌐 在线体验 (Live Demo)

无需配置任何环境，点击下方链接即可直接在浏览器中体验完整的 Web 大屏：
👉 **[点击这里进入在线系统](https://ai-maintenance-system-test-zfmajafiwjabqetfcxzn2t.streamlit.app)**

*(注：如果页面提示休眠，点击唤醒按钮稍等片刻即可)*

---

## 📸 系统界面展示

<img width="959" height="471" alt="image" src="https://github.com/user-attachments/assets/89f33475-3b50-4e9e-be48-d2ea46561de9" />

---

## ✨ 核心功能

*   **多设备动态切换**：支持选择不同设备的运行数据进行实时监控。
*   **AI 异常检测**：底层基于 `Scikit-learn` 的孤立森林无监督学习算法，自动从海量数据中识别异常点。
*   **动态灵敏度调节**：可通过侧边栏滑块实时调整 AI 报警阈值，适应不同工业场景的风险偏好。
*   **智能数据接入**：支持外部 CSV/TXT 数据的即插即用与自动容错解析。
*   **AI 专家诊断 (LLM)**：深度集成智谱 GLM-4.7-Flash 大模型，当检测到异常时，自动生成专业的工业维修建议。
*   **可视化大屏**：使用 `Streamlit` 与 `Matplotlib` 构建，直观展示设备运行趋势与故障预警（红叉标记）。

---

## 🛠️ 技术栈

*   **前端/部署**：Streamlit (构建交互式 Web 界面)
*   **数据处理**：Pandas, NumPy
*   **AI 算法**：Scikit-learn (Isolation Forest)
*   **大模型**：智谱 GLM-4.7-Flash
*   **可视化**：Matplotlib

---

## 💻 本地运行指南

如果你想在本地环境运行该项目，请按照以下步骤操作：

1. 克隆仓库：
   ```bash
   git clone https://github.com/Z-2007/AI-Maintenance-System-2026.git
   ```
2.安装依赖：

   ```bash
   pip install -r requirements.txt
   ```
3.启动应用：

   ```bash
   streamlit run app.py
   ```
## 📊 数据来源

本项目采用 NASA 公开的 C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) 涡扇发动机退化模拟数据集进行算法验证。

---
*Developed for [挑战杯 / 技能竞赛] - 致力于用 AI 赋能工业智能化转型。*
