window.dash_clientside = Object.assign({}, window.dash_clientside, {
    oracle: {
        micHandler: function(nClicks, speakEnabled, currentValue) {
            if (!nClicks) return [currentValue, ""];
            if (!speakEnabled) return [currentValue, "Voice input off"];
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SR) return [currentValue, "SpeechRecognition API unavailable"];
            const rec = new SR();
            rec.lang = "en-US";
            rec.continuous = true;
            rec.interimResults = true;
            return new Promise((resolve) => {
                let finalText = currentValue || "";
                rec.onresult = (event) => {
                    let transcript = "";
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        transcript += event.results[i][0].transcript;
                        if (event.results[i].isFinal) {
                            rec.stop();
                        }
                    }
                    finalText = (finalText + " " + transcript).trim();
                };
                rec.onerror = () => resolve([currentValue, "SpeechRecognition error"]);
                rec.onend = () => resolve([finalText, ""]);
                rec.start();
            });
        },
        ttsHandler: function(history, readEnabled) {
            if (!history || !readEnabled) return "";
            if (!("speechSynthesis" in window)) return "";
            const last = history[history.length - 1];
            if (!last || last.role !== "oracle") return "";
            const text = last.text || "";
            if (!text) return "";
            const u = new SpeechSynthesisUtterance(text);
            u.rate = 0.95;
            u.pitch = 1.0;
            u.volume = 1.0;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
            return "";
        }
    }
});

