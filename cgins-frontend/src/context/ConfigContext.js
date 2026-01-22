"use client";

import { createContext, useContext, useState, useEffect } from "react";

const ConfigContext = createContext();

export const ConfigProvider = ({ children }) => {
    const [config, setConfig] = useState({
        llm_info: {
            model: "",
            apikey: "",
            provider: "anthropic",
        },
    });
    const [loading, setLoading] = useState(true);

    // Load config on mount
    useEffect(() => {
        const fetchConfig = async () => {
            try {
                const res = await fetch("/api/config");
                if (res.ok) {
                    const data = await res.json();
                    // Merge with defaults to ensure structure
                    setConfig((prev) => ({
                        ...prev,
                        ...data,
                        llm_info: { ...prev.llm_info, ...data.llm_info },
                    }));
                }
            } catch (error) {
                console.error("Failed to load config:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchConfig();
    }, []);

    const saveConfig = async (newConfig) => {
        try {
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(newConfig),
            });
            if (res.ok) {
                setConfig(newConfig);
                return true;
            }
            return false;
        } catch (error) {
            console.error("Failed to save config:", error);
            return false;
        }
    };

    return (
        <ConfigContext.Provider value={{ config, setConfig, saveConfig, loading }}>
            {children}
        </ConfigContext.Provider>
    );
};

export const useConfig = () => useContext(ConfigContext);
