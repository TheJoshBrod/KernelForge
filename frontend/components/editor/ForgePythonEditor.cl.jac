"""Forge-themed Monaco editor for code editing/viewing."""

import from react { useEffect, useRef, useState }

def _theme_var(name: str, fallback: str) -> str {
    try {
        root = document.documentElement;
        styles = window.getComputedStyle(root);
        value = styles.getPropertyValue(name);
        if value and value.trim() {
            return value.trim();
        }
    } except Exception {
        return fallback;
    }
    return fallback;
}

def _resolve_editor_language(raw: Any) -> str {
    value = str(raw).toLowerCase() if raw else "python";
    if value == "cuda"
    or value == "c++"
    or value == "cpp"
    or value == "cc"
    or value == "cxx" {
        return "cpp";
    }
    if value == "py" {
        return "python";
    }
    if value == "js" {
        return "javascript";
    }
    if value == "ts" {
        return "typescript";
    }
    if value == "json" {
        return "json";
    }
    if value == "metal" {
        # No dedicated Monaco language id by default; closest syntax
        # for C-like shader code is C++.
        return "cpp";
    }
    if value {
        return value;
    }
    return "python";
}

def _is_alpha(ch: Any) -> bool {
    if not ch or ch.length == 0 {
        return False;
    }
    cp = ch.charCodeAt(0);
    return (cp >= 65 and cp <= 90) or (cp >= 97 and cp <= 122) or ch == "_";
}

def _is_digit(ch: Any) -> bool {
    if not ch or ch.length == 0 {
        return False;
    }
    cp = ch.charCodeAt(0);
    return cp >= 48 and cp <= 57;
}

def _is_ident_char(ch: Any) -> bool {
    return _is_alpha(ch) or _is_digit(ch);
}

def _count_char(text: Any, target: str) -> int {
    if not text {
        return 0;
    }
    c = 0;
    i = 0;
    while i < text.length {
        if text[i] == target {
            c = c + 1;
        }
        i = i + 1;
    }
    return c;
}

def _make_inline_decoration(
    monaco: Any, line_no: int, start_col: int, end_col: int, class_name: str
) -> dict {
    return {
        "range": Reflect.construct(
            monaco.Range, [line_no, start_col, line_no, end_col]
        ),
        "options": {"inlineClassName": class_name}
    };
}

def _build_editor_decorations(monaco: Any, source: Any) -> list {
    text = str(source) if source else "";
    lines = text.split("\n");
    decorations = [];

    in_import_block = False;
    import_depth = 0;

    line_idx = 0;
    while line_idx < lines.length {
        line = lines[line_idx];
        line_no = line_idx + 1;

        # White punctuation for braces/brackets/parentheses.
        i = 0;
        while i < line.length {
            ch = line[i];
            if ch == "("
            or ch == ")"
            or ch == "["
            or ch == "]"
            or ch == "{"
            or ch == "}" {
                decorations.push(
                    _make_inline_decoration(
                        monaco, line_no, i + 1, i + 2, "kf-monaco-inline-punct"
                    )
                );
            }
            i = i + 1;
        }

        # Decorators: color '@' differently from name.
        left = 0;
        while left < line.length and (line[left] == " " or line[left] == "\t") {
            left = left + 1;
        }

        if left < line.length and line[left] == "@" {
            decorations.push(
                _make_inline_decoration(
                    monaco,
                    line_no,
                    left + 1,
                    left + 2,
                    "kf-monaco-inline-decorator-symbol"
                )
            );
            j = left + 1;
            while j < line.length {
                cj = line[j];
                if _is_ident_char(cj) or cj == "." {
                    j = j + 1;
                    continue;
                }
                break;
            }
            if j > left + 1 {
                decorations.push(
                    _make_inline_decoration(
                        monaco,
                        line_no,
                        left + 2,
                        j + 1,
                        "kf-monaco-inline-decorator-name"
                    )
                );
            }
        }

        # Function definitions.
        trim = line.slice(left);
        if trim.startsWith("def ") or trim.startsWith("async def ") {
            def_pos = line.indexOf("def", left);
            if def_pos != -1 {
                k = def_pos + 3;
                while k < line.length and (line[k] == " " or line[k] == "\t") {
                    k = k + 1;
                }
                fn_start = k;
                while k < line.length and _is_ident_char(line[k]) {
                    k = k + 1;
                }
                if k > fn_start {
                    decorations.push(
                        _make_inline_decoration(
                            monaco,
                            line_no,
                            fn_start + 1,
                            k + 1,
                            "kf-monaco-inline-function-name"
                        )
                    );
                }
            }
        }

        # Class definitions.
        if trim.startsWith("class ") {
            class_pos = line.indexOf("class", left);
            if class_pos != -1 {
                k = class_pos + 5;
                while k < line.length and (line[k] == " " or line[k] == "\t") {
                    k = k + 1;
                }
                class_start = k;
                while k < line.length and _is_ident_char(line[k]) {
                    k = k + 1;
                }
                if k > class_start {
                    decorations.push(
                        _make_inline_decoration(
                            monaco,
                            line_no,
                            class_start + 1,
                            k + 1,
                            "kf-monaco-inline-class-name"
                        )
                    );
                }
            }
        }

        # Import statements (including multiline import blocks).
        is_import_line = in_import_block
        or trim.startsWith("import ")
        or trim.startsWith("from ");
        if is_import_line {
            word_start = -1;
            w_i = 0;
            while w_i <= line.length {
                current = line[w_i] if w_i < line.length else " ";
                if _is_ident_char(current) {
                    if word_start == -1 {
                        word_start = w_i;
                    }
                } else {
                    if word_start != -1 {
                        word = line.slice(word_start, w_i);
                        token_class = "";
                        if word == "import" or word == "from" or word == "as" {
                            token_class = "kf-monaco-inline-import-keyword";
                        } elif word[0].toUpperCase() == word[0]
                        and word[0].toLowerCase() != word[0] {
                            token_class = "kf-monaco-inline-type-name";
                        } else {
                            token_class = "kf-monaco-inline-module-name";
                        }
                        decorations.push(
                            _make_inline_decoration(
                                monaco, line_no, word_start + 1, w_i + 1, token_class
                            )
                        );
                        word_start = -1;
                    }
                }
                w_i = w_i + 1;
            }
            depth_delta = _count_char(line, "(") - _count_char(line, ")");
            if not in_import_block and depth_delta > 0 {
                in_import_block = True;
                import_depth = depth_delta;
            } elif in_import_block {
                import_depth = import_depth + depth_delta;
                if import_depth <= 0 {
                    in_import_block = False;
                    import_depth = 0;
                }
            }
        }

        line_idx = line_idx + 1;
    }

    return decorations;
}

def _configure_forge_theme(monaco: Any) -> None {
    if "__kf_monaco_theme_ready" in window and window.__kf_monaco_theme_ready {
        monaco.editor.setTheme("kernal-forge");
        return;
    }

    bg = _theme_var("--bg-darker", "#121417");
    panel = _theme_var("--bg-panel", "#22262b");
    text = _theme_var("--text-light", "#e0e6ed");
    accent_orange = _theme_var("--accent-orange", "#e17055");
    molten = _theme_var("--molten-gold", "#fdcb6e");
    molten_dark = _theme_var("--molten-gold-dark", "#e8a838");
    active_line_bg = _theme_var("--code-active-line-bg", "#fdcb6e1f");
    active_line_border = _theme_var("--code-active-line-border", "#fdcb6e59");
    selection_bg = _theme_var("--code-selection-bg", "#b7544230");
    selection_inactive_bg = _theme_var(
        "--code-selection-inactive-bg", "#8f3f3324"
    );

    monaco.editor.defineTheme(
        "kernal-forge",
        {
            "base": "vs-dark",
            "inherit": True,
            "rules": [
                {"token": "keyword", "foreground": accent_orange[1:]},
                {"token": "keyword.import", "foreground": molten[1:]},
                {"token": "string", "foreground": "ffcf79"},
                {"token": "number", "foreground": "ff9b73"},
                {"token": "comment", "foreground": "8f7567", "fontStyle": "italic"},
                {"token": "operator", "foreground": "e59a72"},
                {"token": "delimiter", "foreground": text[1:]},
                {"token": "delimiter.bracket", "foreground": text[1:]},
                {"token": "type", "foreground": molten_dark[1:]}
            ],
            "colors": {
                "editor.background": bg,
                "editor.foreground": text,
                "editorLineNumber.foreground": "6c7480",
                "editorLineNumber.activeForeground": molten,
                "editorCursor.foreground": molten,
                "editor.selectionBackground": selection_bg,
                "editor.inactiveSelectionBackground": selection_inactive_bg,
                "editor.lineHighlightBackground": active_line_bg,
                "editor.lineHighlightBorder": active_line_border,
                "editorLineNumber.background": panel,
                "editorGutter.background": panel,
                "editorWhitespace.foreground": "ffffff22",
                "editorIndentGuide.background1": "ffffff1a",
                "editorIndentGuide.activeBackground1": accent_orange
            }
        }
    );

    monaco.editor.setTheme("kernal-forge");
    window.__kf_monaco_theme_ready = True;
}

def _load_monaco(on_ready: Any, on_error: Any) -> None {
    if "monaco" in window and window.monaco and "editor" in window.monaco {
        on_ready.call(None, window.monaco);
        return;
    }

    def boot_editor() -> None {
        if not ("require" in window and window.require) {
            on_error.call(None, "Monaco loader unavailable");
            return;
        }
        window.require.config(
            {
                "paths": {
                    "vs": "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.52.2/min/vs"
                }
            }
        );
        window.require(
            ["vs/editor/editor.main"],
            lambda -> None { on_ready.call(None, window.monaco);},
            lambda err: Any  -> None { on_error.call(
                None, err.toString() if err else "Failed to load Monaco editor"
            );}
        );
    }

    if "require" in window and window.require {
        boot_editor();
        return;
    }

    if "__kf_monaco_loader_promise" in window and window.__kf_monaco_loader_promise {
        window.__kf_monaco_loader_promise.then(lambda -> None { boot_editor();}).catch(
            lambda -> None { on_error.call(None, "Failed to load Monaco loader");}
        );
        return;
    }

    def executor(resolve: Any, reject: Any) -> None {
        script = document.createElement("script");
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.52.2/min/vs/loader.min.js";
        script.async = True;
        script.onload = lambda -> None { resolve.call(None, True);};
        script.onerror = lambda -> None { reject.call(
            None, "Failed to load Monaco loader"
        );};
        document.head.appendChild(script);
    }

    window.__kf_monaco_loader_promise = Reflect.construct(Promise, [executor]);
    window.__kf_monaco_loader_promise.then(lambda -> None { boot_editor();}).catch(
        lambda -> None { on_error.call(None, "Failed to load Monaco loader");}
    );
}

def:pub ForgePythonEditor(props: Any) -> Any {
    value = props.value if "value" in props and props.value else "";
    readOnly = props.readOnly if "readOnly" in props and props.readOnly else False;
    language = _resolve_editor_language(
        props.language if "language" in props and props.language else "python"
    );
    placeholder = (
        props.placeholder
        if "placeholder" in props and props.placeholder
        else "# Paste your model code here or upload model.py"
    );
    onChange = props.onChange if "onChange" in props else None;

    container_ref = useRef(None);
    editor_ref = useRef(None);
    monaco_ref = useRef(None);
    latest_value_ref = useRef(value);
    syncing_ref = useRef(False);
    decorations_ref = useRef([]);

    ready_state = useState(False);
    isReady = ready_state[0];
    setIsReady = ready_state[1];

    error_state = useState("");
    loadError = error_state[0];
    setLoadError = error_state[1];

    def applyDecorations() -> None {
        if not editor_ref.current or not monaco_ref.current {
            return;
        }
        current_decorations = (
            decorations_ref.current if decorations_ref.current else []
        );
        if language != "python" {
            decorations_ref.current = editor_ref.current.deltaDecorations(
                current_decorations, []
            );
            return;
        }
        source = editor_ref.current.getValue();
        next_decorations = _build_editor_decorations(monaco_ref.current, source);
        decorations_ref.current = editor_ref.current.deltaDecorations(
            current_decorations, next_decorations
        );
    }

    useEffect(
        lambda -> None { latest_value_ref.current = value if value else "";}, [value]
    );

    useEffect(
        lambda -> Any { disposed = {"value": False};def on_ready(monaco: Any) -> None {
            if disposed["value"] {
                return;
            }
            monaco_ref.current = monaco;
            _configure_forge_theme(monaco);

            host = container_ref.current;
            if not host {
                setLoadError.call(None, "Editor container unavailable");
                return;
            }

            editor = monaco.editor.create(
                host,
                {
                    "value": latest_value_ref.current
                    if latest_value_ref.current
                    else "",
                    "language": language,
                    "theme": "kernal-forge",
                    "automaticLayout": True,
                    "minimap": {"enabled": False},
                    "fontFamily": "JetBrains Mono, Source Code Pro, Menlo, Monaco, Consolas, monospace",
                    "fontSize": 14,
                    "lineHeight": 24,
                    "renderWhitespace": "selection",
                    "scrollBeyondLastLine": False,
                    "wordWrap": "off",
                    "tabSize": 4,
                    "insertSpaces": True,
                    "readOnly": readOnly,
                    "padding": {"top": 18, "bottom": 18},
                    "smoothScrolling": True
                }
            );

            editor_ref.current = editor;
            setLoadError.call(None, "");
            setIsReady.call(None, True);
            applyDecorations();

            editor.onDidChangeModelContent(
                lambda -> None { applyDecorations();if syncing_ref.current {
                    return;
                }if onChange {
                    onChange(editor.getValue());
                }}
            );
        } def on_error(err: Any) -> None {
            if disposed["value"] {
                return;
            }
            setLoadError.call(
                None, str(err) if err else "Unable to initialize IDE editor"
            );
        } _load_monaco(on_ready, on_error);return lambda -> None { disposed["value"] = True;if editor_ref.current {
            editor_ref.current.dispose();
            editor_ref.current = None;
        }decorations_ref.current = [];}; },
        []
    );

    useEffect(
        lambda -> None { if editor_ref.current {
            editor_ref.current.updateOptions({"readOnly": readOnly});
        }},
        [readOnly]
    );

    useEffect(
        lambda -> None { if editor_ref.current and monaco_ref.current {
            model = editor_ref.current.getModel();
            if model {
                monaco_ref.current.editor.setModelLanguage(model, language);
            }
            applyDecorations();
        }},
        [language]
    );

    useEffect(
        lambda -> None { if not editor_ref.current {
            return;
        }incoming = value if value else "";current_value = editor_ref.current.getValue();if current_value != incoming {
            syncing_ref.current = True;
            editor_ref.current.setValue(incoming);
            syncing_ref.current = False;
            applyDecorations();
        }},
        [value]
    );

    fallback_overlay = None;
    if not isReady {
        fallback_overlay = <textarea
            value={value if value else ""}
            onChange={lambda e: Any  -> None { if onChange {
                onChange(e.target.value);
            }}}
            spellCheck={False}
            placeholder={placeholder}
            className="kf-monaco-fallback"
            readOnly={readOnly}
        />;
    }

    error_label = None;
    if loadError {
        error_label = <div className="kf-monaco-error">
            {loadError}
        </div>;
    }

    return
        <div className="kf-monaco-shell">
            <div ref={container_ref} className="kf-monaco-host"></div>
            {fallback_overlay}
            {error_label}
        </div>;
}
