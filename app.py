"""
Interactive Web Application for Deep Learning English Grammatical Error Detection.
Provides a modern Apple Liquid Glass Single-Slide Interface with Clean White Theme,
Floating Micro-Robot Companion, and Rock-Solid Smooth Vertical Scrolling Problem Breakdown Box.

Can be run directly via:
    python3 app.py               # Starts built-in modern Web Dashboard (zero extra dependencies required)
or
    streamlit run app.py         # Runs as a Streamlit application
"""
import sys
import os
import json

# Check if run via Streamlit
if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT"):
    import streamlit as st
    from src.model.detector import DeepGrammarDetector

    st.set_page_config(
        page_title="Deep Learning English Grammar Detector",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Deep Learning Grammatical Error Detection (GED)")
    st.markdown("Transformer Seq2Seq model with Token Alignment & Linguistic Taxonomy Classification")

    @st.cache_resource
    def load_detector():
        return DeepGrammarDetector()

    detector = load_detector()

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["username"] = "Guest Researcher"

    if not st.session_state["authenticated"]:
        st.markdown("""
        <style>
        .stApp { background-color: #fafafa; }
        </style>
        """, unsafe_allow_html=True)
        st.subheader("🔐 Sign In to Workspace")
        st.caption("Deep Learning English Grammatical Error Detection (GED) System")
        with st.form("login_form"):
            uname = st.text_input("Username or Email", value="Harsh Vashisht")
            pwd = st.text_input("Password", type="password", value="••••••••")
            c1, c2 = st.columns(2)
            login_submit = c1.form_submit_button("Sign In to Workspace", type="primary")
            demo_submit = c2.form_submit_button("⚡ Guest Access / Demo")
            if login_submit or demo_submit:
                st.session_state["authenticated"] = True
                st.session_state["username"] = uname if login_submit else "Guest Researcher"
                st.rerun()
        st.stop()

    # Sidebar info
    st.sidebar.markdown(f"**👤 Active User:** `{st.session_state.get('username', 'Harsh Vashisht')}`")
    if st.sidebar.button("🚪 Sign Out"):
        st.session_state["authenticated"] = False
        st.rerun()
    st.sidebar.header("⚙️ Model Architecture")
    st.sidebar.write(f"**Backend:** {detector.backend}")
    st.sidebar.write(f"**Device:** {detector.device.upper()}")
    st.sidebar.write("**Metric Standard:** $F_{0.5}$ (Precision-Weighted)")

    user_input = st.text_area("Enter English Sentence to Analyze:", value="The two freinds is going to the mall togther.", height=110)

    if st.button("✨ Polish & Correct Sentence", type="primary"):
        if user_input.strip():
            result = detector.detect(user_input)

            col1, col2, col3 = st.columns(3)
            col1.metric("Status", "Grammatically Sound" if result.is_grammatically_correct else "Errors Found")
            col2.metric("Error Count", result.error_count)
            col3.metric("Latency", f"{result.processing_time_ms} ms")

            st.subheader("Correction & Analysis")
            if result.is_grammatically_correct:
                st.success("✅ No grammatical errors detected in this sentence!")
            else:
                st.markdown(f"**Suggested Revision:** `{result.corrected_sentence}`")

                for err in result.errors:
                    with st.expander(f"⚠️ [{err.error_type}] '{err.original_text}' ➔ '{err.suggested_text}'", expanded=True):
                        st.write(f"**Explanation:** {err.explanation}")

else:
    # Standalone built-in modern web dashboard server
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse
    import webbrowser
    from src.model.detector import DeepGrammarDetector

    detector = DeepGrammarDetector()

    HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>English Grammar Error Detector</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            /* ⚪ CLEAN APPLE WHITE PALETTE */
            --bg-canvas: #ffffff;
            --card-bg: #ffffff;
            --border-subtle: #e2e8f0;
            --border-strong: #cbd5e1;
            
            --shadow-subtle: 
                0 20px 45px -10px rgba(15, 23, 42, 0.08),
                0 4px 16px -2px rgba(15, 23, 42, 0.04);
                
            --text-main: #0f172a;
            --text-sub: #475569;
            --text-muted: #64748b;
            
            /* Pastel Accents */
            --pastel-purple-bg: #f5f3ff;
            --pastel-purple-border: #ddd6fe;
            --pastel-purple-text: #6d28d9;

            --pastel-mint-bg: #f0fdf4;
            --pastel-mint-border: #bbf7d0;
            --pastel-mint-text: #047857;

            --pastel-rose-bg: #fff1f2;
            --pastel-rose-border: #fecdd3;
            --pastel-rose-text: #be123c;

            --pastel-sky-bg: #f0f9ff;
            --pastel-sky-border: #bae6fd;
            --pastel-sky-text: #0369a1;

            --pastel-amber-bg: #fffbeb;
            --pastel-amber-border: #fde68a;
            --pastel-amber-text: #b45309;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: #f8fafc;
            color: var(--text-main);
            min-height: 100vh;
            width: 100vw;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow-x: hidden;
            overflow-y: auto;
            padding: 30px 20px;
            box-sizing: border-box;
        }

        /* 🪟 SINGLE UNIFIED CLEAN WHITE SLIDE */
        .master-slide {
            width: 100%;
            max-width: 900px;
            background: #ffffff;
            border: 1.5px solid var(--border-subtle);
            border-radius: 28px;
            padding: 26px 32px 26px 32px;
            box-shadow: var(--shadow-subtle);
            position: relative;
            z-index: 10;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            margin: auto;
        }

        /* 🤖 HEADER WITH ANIMATED ROBOT COMPANION */
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border-subtle);
            flex-shrink: 0;
        }

        .header-title-box h1 {
            font-size: 1.85rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.03em;
            display: flex;
            align-items: center;
            gap: 10px;
            line-height: 1.2;
        }
        .header-title-box h1 .title-gradient {
            background: linear-gradient(135deg, #4338ca 0%, #7c3aed 50%, #db2777 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-title-box p {
            color: var(--text-sub);
            font-size: 0.92rem;
            font-weight: 600;
            margin-top: 3px;
        }

        /* CUTE MICRO-ROBOT COMPANION */
        .bot-companion {
            display: flex;
            align-items: center;
            gap: 14px;
            user-select: none;
        }
        .bot-speech-bubble {
            position: relative;
            background: #f8fafc;
            border: 1px solid var(--border-subtle);
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
            border-radius: 16px;
            padding: 8px 14px;
            font-size: 0.84rem;
            font-weight: 700;
            color: #4338ca;
            max-width: 190px;
            line-height: 1.35;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            animation: bubblePulse 4s ease-in-out infinite;
        }
        .bot-speech-bubble::after {
            content: '';
            position: absolute;
            right: -6px;
            top: 50%;
            transform: translateY(-50%) rotate(45deg);
            width: 11px;
            height: 11px;
            background: #f8fafc;
            border-top: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
        }
        @keyframes bubblePulse {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-2px); }
        }

        .bot-figure {
            position: relative;
            width: 58px;
            height: 58px;
            cursor: pointer;
            animation: botHover 3.4s ease-in-out infinite;
        }
        @keyframes botHover {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-7px) rotate(2deg); }
        }

        .bot-shadow {
            position: absolute;
            bottom: -4px;
            left: 14px;
            width: 30px;
            height: 7px;
            background: rgba(79, 70, 229, 0.16);
            border-radius: 50%;
            filter: blur(3px);
            animation: botShadowPulse 3.4s ease-in-out infinite;
        }
        @keyframes botShadowPulse {
            0%, 100% { transform: scale(1); opacity: 0.25; }
            50% { transform: scale(0.75); opacity: 0.12; }
        }

        .bot-svg {
            width: 100%;
            height: 100%;
            filter: drop-shadow(0 4px 10px rgba(99, 102, 241, 0.15));
            transition: transform 0.2s ease;
        }
        .bot-figure:hover .bot-svg {
            transform: scale(1.1) rotate(-4deg);
        }

        .bot-eye {
            transform-origin: center;
            animation: botBlink 3.8s infinite;
        }
        @keyframes botBlink {
            0%, 92%, 100% { transform: scaleY(1); }
            95% { transform: scaleY(0.1); }
        }

        /* Robot thinking mode */
        .bot-companion.thinking .bot-figure {
            animation: botThinkingFloat 1.2s ease-in-out infinite;
        }
        @keyframes botThinkingFloat {
            0%, 100% { transform: translateY(-2px) rotate(-3deg); }
            50% { transform: translateY(-7px) rotate(3deg); }
        }
        .bot-companion.thinking .bot-antenna-light {
            animation: antennaFlash 0.5s ease-in-out infinite alternate !important;
            fill: #a855f7 !important;
        }
        @keyframes antennaFlash {
            from { filter: drop-shadow(0 0 2px #c084fc); }
            to { filter: drop-shadow(0 0 10px #9333ea); }
        }
        .bot-companion.thinking .bot-eyes-idle { display: none !important; }
        .bot-companion.thinking .bot-eyes-thinking { display: block !important; }
        .bot-companion.thinking .laser-scanner {
            animation: laserScan 1s ease-in-out infinite alternate;
        }
        @keyframes laserScan {
            0% { transform: translateX(-10px); }
            100% { transform: translateX(10px); }
        }

        /* Robot happy mode */
        .bot-companion.happy .bot-figure {
            animation: botHappyHop 0.8s ease-in-out 2;
        }
        @keyframes botHappyHop {
            0%, 100% { transform: translateY(0) scale(1); }
            40% { transform: translateY(-12px) scale(1.1); }
            70% { transform: translateY(0) scale(0.95); }
        }
        .bot-companion.happy .bot-eyes-idle { display: none !important; }
        .bot-companion.happy .bot-eyes-thinking { display: none !important; }
        .bot-companion.happy .bot-eyes-happy { display: block !important; }

        /* 📝 INPUT SECTION */
        .input-section {
            flex-shrink: 0;
            margin-bottom: 16px;
        }
        textarea {
            width: 100%;
            height: 90px;
            background: #f8fafc;
            border: 1.5px solid var(--border-subtle);
            border-radius: 18px;
            padding: 14px 18px;
            font-size: 1.06rem;
            line-height: 1.55;
            color: #0f172a;
            font-weight: 600;
            resize: none;
            outline: none;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.02);
            margin-bottom: 12px;
        }
        textarea:focus {
            background: #ffffff;
            border-color: #818cf8;
            box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.18);
        }

        /* CLEAN ACTION BAR - NO PRESETS */
        .action-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .hint-text {
            font-size: 0.86rem;
            color: var(--text-muted);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 7px;
        }
        .hint-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
        }

        /* 🍏 ACTION BUTTON */
        .apple-glass-btn {
            position: relative;
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #818cf8 100%);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 16px;
            color: #ffffff;
            font-size: 0.96rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            padding: 11px 28px;
            box-shadow: 
                0 10px 22px -4px rgba(79, 70, 229, 0.35),
                0 4px 8px -2px rgba(79, 70, 229, 0.2);
            cursor: pointer;
            overflow: hidden;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 9px;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
        }
        .apple-glass-btn:hover {
            background: linear-gradient(135deg, #4338ca 0%, #4f46e5 50%, #6366f1 100%);
            box-shadow: 0 14px 28px -4px rgba(79, 70, 229, 0.45);
            transform: translateY(-1.5px);
        }
        .apple-glass-btn:active {
            transform: translateY(1px) scale(0.98);
        }
        .apple-glass-btn:disabled {
            opacity: 0.85;
            cursor: wait;
            transform: none;
        }

        /* 🌟 OUTPUT SECTION */
        .output-section {
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            animation: fadeIn 0.3s ease-out;
        }

        /* SCANNING / THINKING PANEL */
        .checking-panel {
            padding: 16px 4px;
            animation: fadeIn 0.25s ease-out;
        }
        .checking-glass-banner {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px 20px;
            background: #f8fafc;
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            margin-bottom: 16px;
            position: relative;
            overflow: hidden;
        }
        .checking-glass-banner::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            height: 3px;
            width: 100%;
            background: linear-gradient(90deg, #818cf8, #f472b6, #34d399, #38bdf8, #818cf8);
            background-size: 200% 100%;
            animation: liquidFlow 1.6s linear infinite;
        }
        @keyframes liquidFlow {
            0% { background-position: 100% 0; }
            100% { background-position: -100% 0; }
        }

        .pulse-loader {
            position: relative;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .pulse-glow {
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(129, 140, 248, 0.4) 0%, transparent 70%);
            animation: pulseWave 1.4s infinite cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes pulseWave {
            0% { transform: scale(0.8); opacity: 1; }
            100% { transform: scale(1.8); opacity: 0; }
        }
        .pulse-icon {
            animation: spin 1.2s linear infinite;
            color: #6366f1;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }

        .checking-status-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 3px;
        }
        .checking-status-sub {
            font-size: 0.84rem;
            color: var(--text-sub);
            font-weight: 600;
        }

        /* Shimmer Skeletons */
        .shimmer-card {
            background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%);
            background-size: 200% 100%;
            animation: shimmerSlide 1.5s infinite;
            border-radius: 16px;
        }
        .sk-bar { height: 48px; margin-bottom: 12px; }
        .sk-box { height: 85px; }
        @keyframes shimmerSlide {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        /* STATUS BADGE STRIP */
        .status-header-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 11px 20px;
            border-radius: 16px;
            margin-bottom: 14px;
            font-weight: 700;
            font-size: 0.94rem;
            flex-shrink: 0;
        }
        .status-header-strip.all-clean {
            background: var(--pastel-mint-bg);
            border: 1px solid var(--pastel-mint-border);
            color: var(--pastel-mint-text);
        }
        .status-header-strip.has-errors {
            background: var(--pastel-purple-bg);
            border: 1px solid var(--pastel-purple-border);
            color: var(--pastel-purple-text);
        }
        .status-left {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-pill-count {
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.02em;
        }

        /* 🧊 REVISED TEXT CONTAINER */
        .revised-liquid-box {
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            border: 1.5px solid #a7f3d0;
            border-left: 5px solid #10b981;
            border-radius: 18px;
            padding: 16px 22px;
            margin-bottom: 16px;
            position: relative;
            box-shadow: 0 4px 14px -2px rgba(16, 185, 129, 0.08);
            flex-shrink: 0;
        }
        .revised-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .revised-badge-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 800;
            letter-spacing: 0.07em;
            color: #047857;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .revised-content-text {
            font-size: 1.22rem;
            font-weight: 700;
            color: #064e3b;
            line-height: 1.55;
            padding-right: 85px;
        }

        /* COPY BUTTON */
        .copy-liquid-btn {
            position: absolute;
            top: 14px;
            right: 16px;
            background: #ffffff;
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #047857;
            font-size: 0.82rem;
            font-weight: 800;
            padding: 6px 15px;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.12);
        }
        .copy-liquid-btn:hover {
            background: #f0fdf4;
            transform: translateY(-1.5px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.18);
        }

        /* 📜 VERTICAL SCROLLABLE PROBLEM BREAKDOWN */
        .problems-container {
            display: flex;
            flex-direction: column;
            width: 100%;
        }
        .problems-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 0 4px;
            flex-shrink: 0;
        }
        .problems-heading {
            font-size: 0.95rem;
            font-weight: 800;
            color: #0f172a;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .problems-badge-tag {
            font-size: 0.76rem;
            font-weight: 800;
            background: var(--pastel-purple-bg);
            border: 1px solid var(--pastel-purple-border);
            color: var(--pastel-purple-text);
            padding: 3px 12px;
            border-radius: 16px;
        }

        /* 📜 ROCK-SOLID VERTICAL SCROLL BOX */
        .problems-scroll-box {
            height: 195px;
            max-height: 195px;
            overflow-y: auto;
            overflow-x: hidden;
            overscroll-behavior: contain;
            -webkit-overflow-scrolling: touch;
            padding-right: 8px;
            padding-bottom: 8px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            box-sizing: border-box;
        }
        /* Custom Smooth Visible Scrollbar */
        .problems-scroll-box::-webkit-scrollbar {
            width: 8px;
        }
        .problems-scroll-box::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 10px;
        }
        .problems-scroll-box::-webkit-scrollbar-thumb {
            background: #a78bfa;
            border-radius: 10px;
            border: 2px solid #f1f5f9;
        }
        .problems-scroll-box::-webkit-scrollbar-thumb:hover {
            background: #8b5cf6;
        }

        /* INDIVIDUAL PROBLEM ROW */
        .problem-row-item {
            background: #ffffff;
            border: 1.5px solid var(--border-subtle);
            border-left: 4.5px solid #818cf8;
            border-radius: 16px;
            padding: 12px 18px;
            box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.04);
            transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
            animation: fadeInRow 0.25s ease-out;
            flex-shrink: 0;
        }
        .problem-row-item:hover {
            transform: translateX(4px);
            border-color: #c7d2fe;
            border-left: 5px solid #6366f1;
            box-shadow: 0 6px 18px -2px rgba(99, 102, 241, 0.1);
        }
        @keyframes fadeInRow {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .problem-row-top {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 12px;
            margin-bottom: 6px;
        }

        .taxonomy-badge {
            font-size: 0.74rem;
            font-weight: 800;
            padding: 4px 10px;
            border-radius: 8px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .taxonomy-badge.pastel-purple {
            background: var(--pastel-purple-bg);
            color: var(--pastel-purple-text);
            border: 1px solid var(--pastel-purple-border);
        }
        .taxonomy-badge.pastel-rose {
            background: var(--pastel-rose-bg);
            color: var(--pastel-rose-text);
            border: 1px solid var(--pastel-rose-border);
        }
        .taxonomy-badge.pastel-amber {
            background: var(--pastel-amber-bg);
            color: var(--pastel-amber-text);
            border: 1px solid var(--pastel-amber-border);
        }
        .taxonomy-badge.pastel-sky {
            background: var(--pastel-sky-bg);
            color: var(--pastel-sky-text);
            border: 1px solid var(--pastel-sky-border);
        }

        /* Token Diff Flow */
        .token-flow-bar {
            display: inline-flex;
            align-items: center;
            gap: 9px;
            font-size: 1.02rem;
            font-weight: 700;
        }
        .orig-word {
            background: #fee2e2;
            color: #991b1b;
            padding: 3px 9px;
            border-radius: 7px;
            font-weight: 800;
            text-decoration: line-through;
        }
        .flow-arrow {
            color: #94a3b8;
            font-size: 0.9rem;
            font-weight: 800;
        }
        .sugg-word {
            background: #dcfce7;
            color: #166534;
            padding: 3px 9px;
            border-radius: 7px;
            font-weight: 800;
        }

        .problem-desc {
            font-size: 0.90rem;
            color: #334155;
            line-height: 1.48;
            font-weight: 600;
        }

        /* SUCCESS STATE */
        .all-clean-card {
            text-align: center;
            padding: 24px 10px;
        }
        .all-clean-icon {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: var(--pastel-mint-bg);
            border: 1.5px solid var(--pastel-mint-border);
            color: var(--pastel-mint-text);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 10px;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
        }
        .all-clean-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: #065f46;
            margin-bottom: 4px;
        }
        .all-clean-desc {
            font-size: 0.92rem;
            color: var(--text-sub);
            font-weight: 600;
        }

        /* ⚠️ HINGLISH LANGUAGE MISMATCH ALERT CARD */
        .hinglish-alert-card {
            background: #fffbeb;
            border: 1.5px solid #fde68a;
            border-radius: 18px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 4px 14px rgba(245, 158, 11, 0.08);
            animation: fadeIn 0.25s ease-out;
            text-align: left;
        }
        .hinglish-alert-badge {
            display: inline-block;
            background: #fef3c7;
            color: #b45309;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 3px 8px;
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .hinglish-alert-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #92400e;
            margin-bottom: 6px;
        }
        .hinglish-alert-desc {
            font-size: 0.90rem;
            color: #b45309;
            line-height: 1.5;
            font-weight: 500;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* =========================================================
           🔐 FRONT LOGIN SCREEN: APPLE LIQUID GLASS CARD STYLING
           ========================================================= */
        .auth-wrapper {
            width: 100%;
            max-width: 440px;
            margin: auto;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 24px;
            padding: 36px 32px 32px 32px;
            box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.08), 0 0 0 1px rgba(0, 0, 0, 0.03);
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            animation: fadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1);
            box-sizing: border-box;
            position: relative;
            z-index: 10;
        }

        .auth-logo-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(219, 39, 119, 0.08));
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 999px;
            color: #4f46e5;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .auth-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.03em;
            margin-bottom: 6px;
            line-height: 1.2;
        }

        .auth-subtitle {
            font-size: 0.88rem;
            color: #64748b;
            margin-bottom: 22px;
            line-height: 1.5;
        }

        .auth-tabs {
            display: flex;
            width: 100%;
            background: #f1f5f9;
            border-radius: 12px;
            padding: 4px;
            margin-bottom: 20px;
            box-sizing: border-box;
        }

        .auth-tab {
            flex: 1;
            padding: 8px 14px;
            border: none;
            background: transparent;
            border-radius: 9px;
            font-size: 13px;
            font-weight: 600;
            color: #64748b;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .auth-tab.active {
            background: #ffffff;
            color: #0f172a;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        }

        .auth-form {
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 14px;
            text-align: left;
            box-sizing: border-box;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
            width: 100%;
            box-sizing: border-box;
        }

        .form-label {
            font-size: 11.5px;
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-container {
            position: relative;
            display: flex;
            align-items: center;
            width: 100%;
            box-sizing: border-box;
        }

        .input-icon {
            position: absolute;
            left: 13px;
            color: #94a3b8;
            pointer-events: none;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .auth-input {
            width: 100%;
            padding: 12px 14px 12px 42px;
            border: 1.5px solid #e2e8f0;
            border-radius: 12px;
            font-size: 14px;
            font-family: inherit;
            color: #0f172a;
            background: #f8fafc;
            transition: all 0.2s;
            outline: none;
            box-sizing: border-box;
        }

        .auth-input:focus {
            background: #ffffff;
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }

        .toggle-password {
            position: absolute;
            right: 12px;
            background: none;
            border: none;
            color: #94a3b8;
            cursor: pointer;
            font-size: 15px;
            padding: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .auth-remember-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12.5px;
            color: #475569;
            margin-top: 2px;
            width: 100%;
        }

        .auth-remember-label {
            display: flex;
            align-items: center;
            gap: 7px;
            cursor: pointer;
            user-select: none;
        }

        .auth-remember-label input[type="checkbox"] {
            accent-color: #4f46e5;
            cursor: pointer;
        }

        .auth-btn-primary {
            width: 100%;
            padding: 13px 20px;
            background: linear-gradient(135deg, #4338ca 0%, #6366f1 50%, #7c3aed 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-size: 14.5px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
            margin-top: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-sizing: border-box;
        }

        .auth-btn-primary:hover {
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.42);
            transform: translateY(-1px);
        }

        .auth-btn-demo {
            width: 100%;
            padding: 11px 20px;
            background: #ffffff;
            color: #334155;
            border: 1.5px solid #e2e8f0;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-sizing: border-box;
        }

        .auth-btn-demo:hover {
            background: #f1f5f9;
            border-color: #cbd5e1;
            color: #0f172a;
        }

        .auth-features-list {
            margin-top: 22px;
            padding-top: 18px;
            border-top: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            gap: 8px;
            width: 100%;
            text-align: left;
            font-size: 11.5px;
            color: #64748b;
            box-sizing: border-box;
        }

        .auth-feature-item {
            display: flex;
            align-items: center;
            gap: 8px;
            line-height: 1.4;
        }

        /* 👤 USER PROFILE BADGE IN DASHBOARD */
        .user-profile-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 5px 12px 5px 6px;
            background: #f8fafc;
            border: 1px solid var(--border-subtle);
            border-radius: 999px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            margin-top: 8px;
        }

        .user-avatar {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: #ffffff;
            font-size: 11px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .user-meta {
            display: flex;
            flex-direction: column;
            line-height: 1.15;
            text-align: left;
        }

        .user-name {
            font-size: 12px;
            font-weight: 700;
            color: #0f172a;
        }

        .user-status-dot {
            font-size: 10px;
            font-weight: 600;
            color: #10b981;
        }

        .signout-btn {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            padding: 3px 9px;
            font-size: 11px;
            font-weight: 600;
            color: #64748b;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: all 0.2s;
            margin-left: 4px;
        }

        .signout-btn:hover {
            color: #ef4444;
            border-color: #fecaca;
            background: #fff5f5;
        }
    </style>
</head>
<body>
    <!-- 🔐 FRONT LOGIN SCREEN (DEFAULT VISIBLE) -->
    <div class="auth-wrapper" id="loginView" style="display: flex;">
        <div class="auth-logo-badge">
            <span>✨ DeepGrammar AI</span>
        </div>
        <h2 class="auth-title">Welcome Back</h2>
        <p class="auth-subtitle">Sign in to access real-time NLP error detection & deep sequence-to-sequence grammar analytics.</p>
        
        <div class="auth-tabs">
            <button class="auth-tab active" type="button" onclick="setAuthTab('login')">Sign In</button>
            <button class="auth-tab" type="button" onclick="setAuthTab('register')">Register</button>
        </div>

        <form class="auth-form" onsubmit="handleLogin(event)">
            <div class="form-group">
                <label class="form-label" for="loginUsername">Username / Email</label>
                <div class="input-container">
                    <span class="input-icon">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                    </span>
                    <input type="text" class="auth-input" id="loginUsername" placeholder="e.g. Harsh Vashisht" value="Harsh Vashisht" required autocomplete="username">
                </div>
            </div>

            <div class="form-group">
                <label class="form-label" for="loginPassword">Password</label>
                <div class="input-container">
                    <span class="input-icon">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
                    </span>
                    <input type="password" class="auth-input" id="loginPassword" value="password123" required autocomplete="current-password">
                    <button type="button" class="toggle-password" id="togglePwdBtn" onclick="togglePasswordVisibility()" title="Toggle Password">👁️</button>
                </div>
            </div>

            <div class="auth-remember-row">
                <label class="auth-remember-label">
                    <input type="checkbox" id="rememberMe" checked>
                    <span>Keep me signed in</span>
                </label>
                <a href="javascript:void(0)" onclick="handleGuestLogin()" style="color: #6366f1; text-decoration: none; font-weight: 600;">Fast Demo ➔</a>
            </div>

            <button type="submit" class="auth-btn-primary">
                <span>Sign In to Workspace</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
            </button>

            <button type="button" class="auth-btn-demo" onclick="handleGuestLogin()">
                <span>⚡ Guest Access (One-Click Demo)</span>
            </button>
        </form>

        <div class="auth-features-list">
            <div class="auth-feature-item">
                <span style="color: #10b981;">✓</span>
                <span><b>Zero-Cost Local Inference</b> — No cloud API bills or external keys</span>
            </div>
            <div class="auth-feature-item">
                <span style="color: #6366f1;">✓</span>
                <span><b>Transformer Seq2Seq</b> — Deep syntactic & semantic error correction</span>
            </div>
            <div class="auth-feature-item">
                <span style="color: #f59e0b;">✓</span>
                <span><b>Sub-millisecond Pipeline</b> — SymSpell O(1) + Trigram Perplexity filters</span>
            </div>
        </div>
    </div>

    <!-- 🪟 SINGLE UNIFIED CLEAN WHITE SLIDE -->
    <div class="master-slide" id="mainAppView" style="display: none;">
        <!-- 🤖 COMPACT HEADER WITH ANIMATED ROBOT COMPANION -->
        <header>
            <div class="header-title-box">
                <h1>
                    <span>English Grammar</span>
                    <span class="title-gradient">Detector</span>
                </h1>
                <p>Deep Learning GED & Real-Time Linguistic Error Correction</p>
                <div class="user-profile-badge">
                    <div class="user-avatar" id="userAvatar">HV</div>
                    <div class="user-meta">
                        <span class="user-name" id="userNameLabel">Harsh Vashisht</span>
                        <span class="user-status-dot">● Active Session</span>
                    </div>
                    <button class="signout-btn" onclick="handleSignOut()" title="Sign Out">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>
                        <span>Sign Out</span>
                    </button>
                </div>
            </div>

            <!-- MICRO-ROBOT COMPANION -->
            <div class="bot-companion" id="botCompanion">
                <div class="bot-speech-bubble" id="botSpeech">Type anything to polish! ✨</div>
                <div class="bot-figure" onclick="pokeRobot()" title="Say hello to your grammar robot!">
                    <svg class="bot-svg" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <!-- Antenna -->
                        <path d="M50 28 V15" stroke="#94a3b8" stroke-width="3.5" stroke-linecap="round"/>
                        <circle class="bot-antenna-light" cx="50" cy="12" r="5" fill="#818cf8"/>
                        
                        <!-- Head / Visor Shell -->
                        <rect x="22" y="26" width="56" height="42" rx="16" fill="url(#botHeadGrad)" stroke="#e2e8f0" stroke-width="2"/>
                        
                        <!-- Glass Visor Screen -->
                        <rect class="bot-visor" x="28" y="32" width="44" height="28" rx="10" fill="#0f172a"/>
                        
                        <!-- Idle Eyes (Dots) -->
                        <g class="bot-eyes-idle">
                            <circle class="bot-eye left" cx="40" cy="46" r="4.2" fill="#38bdf8"/>
                            <circle class="bot-eye right" cx="60" cy="46" r="4.2" fill="#38bdf8"/>
                        </g>
                        
                        <!-- Thinking Eyes (Scanning Laser) -->
                        <g class="bot-eyes-thinking" style="display: none;">
                            <rect class="laser-scanner" x="42" y="44" width="16" height="4" rx="2" fill="#818cf8"/>
                        </g>
                        
                        <!-- Happy Eyes (Curved Arcs) -->
                        <g class="bot-eyes-happy" style="display: none;">
                            <path d="M35 48 Q40 41 45 48" stroke="#34d399" stroke-width="3" stroke-linecap="round" fill="none"/>
                            <path d="M55 48 Q60 41 65 48" stroke="#34d399" stroke-width="3" stroke-linecap="round" fill="none"/>
                        </g>

                        <!-- Little Cheek Blush -->
                        <ellipse class="bot-blush" cx="33" cy="54" rx="3.5" ry="2" fill="#f43f5e" opacity="0.35"/>
                        <ellipse class="bot-blush" cx="67" cy="54" rx="3.5" ry="2" fill="#f43f5e" opacity="0.35"/>

                        <!-- Neck & Floating Pod -->
                        <rect x="42" y="68" width="16" height="6" rx="3" fill="#cbd5e1"/>
                        <path d="M36 74 C36 74, 42 86, 50 86 C58 86, 64 74, 64 74 Z" fill="url(#botBaseGrad)"/>
                        <ellipse class="bot-jet-glow" cx="50" cy="85" rx="7" ry="2.5" fill="#38bdf8" opacity="0.7"/>

                        <defs>
                            <linearGradient id="botHeadGrad" x1="22" y1="26" x2="78" y2="68" gradientUnits="userSpaceOnUse">
                                <stop stop-color="#ffffff"/>
                                <stop offset="1" stop-color="#f8fafc"/>
                            </linearGradient>
                            <linearGradient id="botBaseGrad" x1="36" y1="74" x2="64" y2="86" gradientUnits="userSpaceOnUse">
                                <stop stop-color="#e2e8f0"/>
                                <stop offset="1" stop-color="#94a3b8"/>
                            </linearGradient>
                        </defs>
                    </svg>
                    <div class="bot-shadow"></div>
                </div>
            </div>
        </header>

        <!-- 📝 INPUT SECTION (ZERO PRESETS UNDERNEATH) -->
        <div class="input-section">
            <textarea id="sentenceInput" placeholder="Type or paste your English sentence here...">The two freinds is going to the mall togther.</textarea>
            <div class="action-bar">
                <div class="hint-text">
                    <span class="hint-dot"></span>
                    <span>NLP & Statistical Grammar Engine Active</span>
                </div>
                <!-- 🍏 ACTION BUTTON -->
                <button class="apple-glass-btn" id="checkBtn" onclick="analyzeSentence()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="11" cy="11" r="8"></circle>
                        <path d="m21 21-4.35-4.35"></path>
                    </svg>
                    <span>Check Grammar</span>
                </button>
            </div>
        </div>

        <!-- 🌟 OUTPUT SECTION (INSIDE SAME MASTER SLIDE) -->
        <div id="outputSection" class="output-section">
            
            <!-- 🌀 THINKING / SCANNING STATE -->
            <div id="checkingPanel" class="checking-panel" style="display: none;">
                <div class="checking-glass-banner">
                    <div class="pulse-loader">
                        <div class="pulse-glow"></div>
                        <svg class="pulse-icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                            <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
                        </svg>
                    </div>
                    <div>
                        <div class="checking-status-title" id="checkingStatusTitle">Analyzing sentence syntax & structure...</div>
                        <div class="checking-status-sub">Transformer Seq2Seq + Statistical Lexical Knowledge Engine</div>
                    </div>
                </div>

                <!-- Shimmer Skeletons -->
                <div class="shimmer-card sk-bar"></div>
                <div class="shimmer-card sk-box"></div>
            </div>

            <!-- 🌟 FINAL RESULTS CONTAINER -->
            <div id="resultsContent" style="display: none; flex-direction: column;">
                
                <!-- Status Strip -->
                <div id="statusStrip" class="status-header-strip">
                    <div class="status-left">
                        <span id="statusIcon">✨</span>
                        <span id="statusMainText">Sentence Polished</span>
                    </div>
                    <div class="status-pill-count" id="statusCountPill">3 Issues Resolved</div>
                </div>

                <!-- ⚠️ Hinglish Warning Notice Card (English Only) -->
                <div id="hinglishAlertCard" class="hinglish-alert-card" style="display: none;">
                    <div class="hinglish-alert-badge">Notice</div>
                    <div class="hinglish-alert-title">⚠️ Non-English Language Detected</div>
                    <div class="hinglish-alert-desc">
                        This system is specifically built for English Grammatical Error Detection. Please enter your sentence in standard English only.
                    </div>
                </div>

                <!-- Revised Text Container -->
                <div id="revisedContainer" class="revised-liquid-box" style="display: none;">
                    <div class="revised-header-row">
                        <span class="revised-badge-label">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                            <span>Corrected Sentence</span>
                        </span>
                        <button class="copy-liquid-btn" onclick="copyCorrection()">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                            <span id="copyBtnLabel">Copy</span>
                        </button>
                    </div>
                    <div class="revised-content-text" id="correctedText"></div>
                </div>

                <!-- All Clean Box (When no errors found) -->
                <div id="allCleanBox" class="all-clean-card" style="display: none;">
                    <div class="all-clean-icon">
                        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                    </div>
                    <div class="all-clean-title">Grammatically Sound!</div>
                    <div class="all-clean-desc">No grammatical or spelling mistakes detected in this sentence.</div>
                </div>

                <!-- 📜 VERTICAL SCROLLABLE PROBLEM BREAKDOWN -->
                <div id="problemsSection" class="problems-container" style="display: none;">
                    <div class="problems-header-row">
                        <span class="problems-heading">
                            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2.3"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                            <span>Problem Breakdown</span>
                        </span>
                        <span class="problems-badge-tag" id="problemsBadgeTag">3 Detected Issues</span>
                    </div>

                    <!-- VERTICAL SCROLL BOX -->
                    <div class="problems-scroll-box" id="problemsScrollBox"></div>
                </div>

            </div>

        </div>
    </div>

    <script>
        const clientCache = new Map();
        let checkingInterval = null;

        const thinkingPhrases = [
            "Analyzing sentence syntax & structure...",
            "Validating dictionary, spelling & homophones...",
            "Checking subject-verb agreement & tenses...",
            "Synthesizing deep learning corrections..."
        ];

        const botThinkingLines = [
            "Scanning grammar rules...",
            "Checking spelling & typos...",
            "Harmonizing verbs & tenses...",
            "Polishing sentence..."
        ];

        function pokeRobot() {
            const bot = document.getElementById('botCompanion');
            const speech = document.getElementById('botSpeech');
            bot.classList.add('happy');
            speech.innerText = "I love helping you write! 😊";
            setTimeout(() => {
                bot.classList.remove('happy');
                speech.innerText = "Ready to polish! ✨";
            }, 2500);
        }

        async function analyzeSentence() {
            const input = document.getElementById('sentenceInput').value.trim();
            if (!input) return;

            const btn = document.getElementById('checkBtn');
            const checkingPanel = document.getElementById('checkingPanel');
            const resultsContent = document.getElementById('resultsContent');
            const checkingStatusTitle = document.getElementById('checkingStatusTitle');
            const bot = document.getElementById('botCompanion');
            const botSpeech = document.getElementById('botSpeech');

            // 1. Switch Robot to Thinking Mode
            bot.classList.remove('happy');
            bot.classList.add('thinking');
            botSpeech.innerText = botThinkingLines[0];

            // 2. Button in Loading State
            btn.disabled = true;
            btn.innerHTML = `
                <svg class="pulse-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
                </svg>
                <span>Analyzing...</span>
            `;

            // 3. Show Checking Animation inside single output card
            resultsContent.style.display = 'none';
            checkingPanel.style.display = 'block';

            let stepIdx = 0;
            checkingStatusTitle.innerText = thinkingPhrases[0];
            if (checkingInterval) clearInterval(checkingInterval);
            checkingInterval = setInterval(() => {
                stepIdx = (stepIdx + 1) % thinkingPhrases.length;
                checkingStatusTitle.innerText = thinkingPhrases[stepIdx];
                botSpeech.innerText = botThinkingLines[stepIdx];
            }, 320);

            // Fast-path client cache check
            if (clientCache.has(input)) {
                setTimeout(() => {
                    finishAnalysis(clientCache.get(input));
                }, 220);
                return;
            }

            try {
                const t0 = performance.now();
                const res = await fetch('/api/detect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({sentence: input})
                });
                const data = await res.json();
                clientCache.set(input, data);

                const elapsed = performance.now() - t0;
                const minWait = Math.max(0, 300 - elapsed);
                setTimeout(() => {
                    finishAnalysis(data);
                }, minWait);

            } catch(e) {
                clearInterval(checkingInterval);
                resetButton();
                bot.classList.remove('thinking');
                botSpeech.innerText = "Oops, something went wrong!";
                alert("Error connecting to server: " + e);
            }
        }

        function finishAnalysis(data) {
            clearInterval(checkingInterval);
            resetButton();

            const bot = document.getElementById('botCompanion');
            const botSpeech = document.getElementById('botSpeech');
            
            // Switch bot to happy celebratory mode
            bot.classList.remove('thinking');
            bot.classList.add('happy');
            const isHinglish = data.is_hinglish || (data.errors && data.errors.some(e => e.error_type && e.error_type.includes("Hinglish")));
            if (isHinglish) {
                botSpeech.innerText = "Please write only English! ⚠️";
            } else if (data.is_grammatically_correct) {
                botSpeech.innerText = "Clean & flawless! 🎉";
            } else {
                botSpeech.innerText = `Fixed ${data.error_count} issue${data.error_count > 1 ? 's' : ''}! ✨`;
            }
            setTimeout(() => {
                bot.classList.remove('happy');
            }, 3000);

            renderResults(data);
        }

        function resetButton() {
            const btn = document.getElementById('checkBtn');
            btn.disabled = false;
            btn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <path d="m21 21-4.35-4.35"></path>
                </svg>
                <span>Check Grammar</span>
            `;
        }

        function renderResults(data) {
            const checkingPanel = document.getElementById('checkingPanel');
            const resultsContent = document.getElementById('resultsContent');
            const statusStrip = document.getElementById('statusStrip');
            const statusIcon = document.getElementById('statusIcon');
            const statusMainText = document.getElementById('statusMainText');
            const statusCountPill = document.getElementById('statusCountPill');
            const revisedContainer = document.getElementById('revisedContainer');
            const correctedText = document.getElementById('correctedText');
            const allCleanBox = document.getElementById('allCleanBox');
            const problemsSection = document.getElementById('problemsSection');
            const problemsScrollBox = document.getElementById('problemsScrollBox');
            const problemsBadgeTag = document.getElementById('problemsBadgeTag');
            const hinglishAlertCard = document.getElementById('hinglishAlertCard');

            checkingPanel.style.display = 'none';
            resultsContent.style.display = 'flex';

            const isHinglish = data.is_hinglish || (data.errors && data.errors.some(e => e.error_type && e.error_type.includes("Hinglish")));

            if (isHinglish) {
                statusStrip.className = "status-header-strip has-errors";
                statusStrip.style.background = "#fffbeb";
                statusStrip.style.borderColor = "#f59e0b";
                statusIcon.innerText = "⚠️";
                statusMainText.innerText = "Non-English Language Detected";
                statusCountPill.innerText = "English Only";
                statusCountPill.style.background = "#fef3c7";
                statusCountPill.style.color = "#b45309";

                if (hinglishAlertCard) hinglishAlertCard.style.display = 'block';
                allCleanBox.style.display = 'none';
                revisedContainer.style.display = 'none';
                problemsSection.style.display = 'none';
            } else if (data.is_grammatically_correct) {
                if (hinglishAlertCard) hinglishAlertCard.style.display = 'none';
                statusStrip.className = "status-header-strip all-clean";
                statusStrip.style.background = "";
                statusStrip.style.borderColor = "";
                statusIcon.innerText = "✅";
                statusMainText.innerText = "Flawless English — No Errors Found";
                statusCountPill.innerText = "0 Errors";
                statusCountPill.style.background = "";
                statusCountPill.style.color = "";
                
                revisedContainer.style.display = 'none';
                problemsSection.style.display = 'none';
                allCleanBox.style.display = 'block';
            } else {
                if (hinglishAlertCard) hinglishAlertCard.style.display = 'none';
                statusStrip.className = "status-header-strip has-errors";
                statusStrip.style.background = "";
                statusStrip.style.borderColor = "";
                statusIcon.innerText = "✨";
                statusMainText.innerText = "Polished & Corrected";
                statusCountPill.innerText = `${data.error_count} ${data.error_count > 1 ? 'Issues' : 'Issue'} Resolved`;
                statusCountPill.style.background = "";
                statusCountPill.style.color = "";

                allCleanBox.style.display = 'none';
                revisedContainer.style.display = 'block';
                correctedText.innerText = data.corrected_sentence || data.original_sentence;
            }

            if (!isHinglish && !data.is_grammatically_correct && data.errors && data.errors.length > 0) {
                // Render Vertically Scrollable Problem Items
                problemsSection.style.display = 'flex';
                problemsBadgeTag.innerText = `${data.errors.length} Detected ${data.errors.length > 1 ? 'Issues' : 'Issue'}`;

                const badgeColors = ['pastel-purple', 'pastel-rose', 'pastel-amber', 'pastel-sky'];
                let listHtml = '';

                data.errors.forEach((err, idx) => {
                    const orig = err.original_text || '[missing]';
                    const sugg = err.suggested_text || '[omitted]';
                    const colorClass = badgeColors[idx % badgeColors.length];

                    listHtml += `
                        <div class="problem-row-item">
                            <div class="problem-row-top">
                                <span class="taxonomy-badge ${colorClass}">${err.error_type}</span>
                                <div class="token-flow-bar">
                                    <span class="orig-word">${orig}</span>
                                    <span class="flow-arrow">➔</span>
                                    <span class="sugg-word">${sugg}</span>
                                </div>
                            </div>
                            <div class="problem-desc">${err.explanation}</div>
                        </div>
                    `;
                });

                problemsScrollBox.innerHTML = listHtml;
                problemsScrollBox.scrollTop = 0;
            }
        }

        function copyCorrection() {
            const txt = document.getElementById('correctedText').innerText;
            navigator.clipboard.writeText(txt);
            const label = document.getElementById('copyBtnLabel');
            label.innerText = 'Copied! ✨';
            setTimeout(() => { label.innerText = 'Copy'; }, 1800);
        }


        // --- 🔐 AUTHENTICATION LOGIC ---
        // Guarantees that opening the link always displays the Front Login Gate first!
        function checkAuthState() {
            const user = sessionStorage.getItem('ged_user');
            const loginView = document.getElementById('loginView');
            const mainView = document.getElementById('mainAppView');
            const nameLabel = document.getElementById('userNameLabel');
            const avatarLabel = document.getElementById('userAvatar');

            if (user) {
                if (loginView) loginView.style.display = 'none';
                if (mainView) {
                    mainView.style.display = 'flex';
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                }
                if (nameLabel) nameLabel.innerText = user;
                if (avatarLabel) {
                    const initials = user.split(' ').filter(Boolean).map(n => n[0]).join('').substring(0, 2).toUpperCase() || 'U';
                    avatarLabel.innerText = initials;
                }
                try {
                    analyzeSentence();
                } catch(err) {
                    console.error("Auto analyze error:", err);
                }
            } else {
                if (loginView) loginView.style.display = 'flex';
                if (mainView) mainView.style.display = 'none';
            }
        }

        function handleLogin(e) {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            const usernameInput = document.getElementById('loginUsername');
            const username = (usernameInput && usernameInput.value.trim()) || 'Harsh Vashisht';
            sessionStorage.setItem('ged_user', username);
            checkAuthState();
            return false;
        }

        function handleGuestLogin(e) {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            sessionStorage.setItem('ged_user', 'Guest Researcher');
            checkAuthState();
            return false;
        }

        function handleSignOut() {
            sessionStorage.removeItem('ged_user');
            try { localStorage.removeItem('ged_user'); } catch(e){}
            checkAuthState();
        }

        function togglePasswordVisibility() {
            const pwd = document.getElementById('loginPassword');
            const btn = document.getElementById('togglePwdBtn');
            if (pwd.type === 'password') {
                pwd.type = 'text';
                btn.innerText = '🙈';
            } else {
                pwd.type = 'password';
                btn.innerText = '👁️';
            }
        }

        function setAuthTab(tab) {
            const tabs = document.querySelectorAll('.auth-tab');
            tabs.forEach(t => t.classList.remove('active'));
            const titleEl = document.querySelector('.auth-title');
            const subTitleEl = document.querySelector('.auth-subtitle');
            const btnSpan = document.querySelector('.auth-btn-primary span');

            if (tab === 'login') {
                if (tabs[0]) tabs[0].classList.add('active');
                if (titleEl) titleEl.innerText = 'Welcome Back';
                if (subTitleEl) subTitleEl.innerText = 'Sign in to access real-time NLP error detection & deep sequence-to-sequence grammar analytics.';
                if (btnSpan) btnSpan.innerText = 'Sign In to Workspace';
            } else {
                if (tabs[1]) tabs[1].classList.add('active');
                if (titleEl) titleEl.innerText = 'Create Account';
                if (subTitleEl) subTitleEl.innerText = 'Sign up for instant free access to grammar correction & analytics.';
                if (btnSpan) btnSpan.innerText = 'Sign Up & Enter Workspace';
            }
        }

        // Run once on initial page load
        window.addEventListener('DOMContentLoaded', () => {
            try { localStorage.removeItem('ged_user'); } catch(e){}
            checkAuthState();
        });
    </script>
</body>
</html>
"""

    class DashboardHandler(BaseHTTPRequestHandler):
        def _set_headers(self, status=200, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
            self.end_headers()

        def do_OPTIONS(self):
            self._set_headers(200)

        def do_HEAD(self):
            self._set_headers(200, content_type="text/html")

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/" or parsed.path == "/index.html":
                self._set_headers(200, content_type="text/html")
                self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
            # Report serving removed for clean standalone app
            elif parsed.path == "/status":
                self._set_headers(200)
                resp = {"model_backend": detector.backend, "device": detector.device}
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            else:
                self._set_headers(404)
                self.wfile.write(b'{"error": "Not found"}')

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/detect":
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len)
                try:
                    payload = json.loads(post_body.decode("utf-8"))
                    sentence = payload.get("sentence", "")
                    result = detector.detect(sentence)
                    self._set_headers(200)
                    self.wfile.write(json.dumps(result.to_dict()).encode("utf-8"))
                except Exception as e:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            else:
                self._set_headers(404)
                self.wfile.write(b'{"error": "Not found"}')

    def run_web_dashboard(port: int = 8080):
        print("=" * 65)
        print("  English Grammatical Error Detection (GED) Web Dashboard")
        print("=" * 65)
        print(f"Server running at: http://localhost:{port}")
        print("Press Ctrl+C to stop the dashboard.\n")
        HTTPServer.allow_reuse_address = True
        server = HTTPServer(("0.0.0.0", port), DashboardHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8080, help="Port to run web app (default: 8080)")
        args = parser.parse_args()
        run_web_dashboard(port=args.port)
