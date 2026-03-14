"""Searchable Select Component"""

import from react { useState, useEffect, useRef }
import from "react-dom" { createPortal }
import from .Icons { ChevronDownIcon, SearchIcon, CheckIcon }

def:pub SearchableSelect(props: any) -> Any {
    # Props: value, onChange, options, placeholder, disabled, loading
    value = props.value;
    onChange = props.onChange;
    options = props.options;
    placeholder = props.placeholder;
    disabled = props.disabled;
    loading = props.loading;

    isOpenState = useState(False);
    isOpen = isOpenState[0];
    setIsOpen = isOpenState[1];

    searchTermState = useState("");
    searchTerm = searchTermState[0];
    setSearchTerm = searchTermState[1];

    wrapperRef = useRef(None);
    portalRef = useRef(None);
    coordsState = useState({"top": 0, "left": 0, "width": 0});
    coords = coordsState[0];
    setCoords = coordsState[1];

    def updateCoords -> None {
        if wrapperRef.current {
            rect = wrapperRef.current.getBoundingClientRect();
            setCoords({
                "top": rect.bottom,
                "left": rect.left,
                "width": rect.width
            });
        }
    }

    useEffect(
        lambda -> None {
            if isOpen {
                updateCoords();
                window.addEventListener("scroll", updateCoords, True);
                window.addEventListener("resize", updateCoords);
            }
            return lambda -> None {
                window.removeEventListener("scroll", updateCoords, True);
                window.removeEventListener("resize", updateCoords);
            };
        },
        [isOpen]
    );

    # Filter options
    filteredOptions = options.filter(
        lambda (o: any) -> bool { if not searchTerm {
            return True;
        }return o["label"].toLowerCase().includes(searchTerm.toLowerCase())
        or o["value"].toLowerCase().includes(searchTerm.toLowerCase()); }
    );

    # Handle outside click to close
    useEffect(
        lambda -> Any { def handleClickOutside(event: any) -> None {
            in_wrapper = wrapperRef.current and wrapperRef.current.contains(event.target);
            in_portal = portalRef.current and portalRef.current.contains(event.target);
            if not in_wrapper and not in_portal {
                setIsOpen(False);
            }
        } document.addEventListener("mousedown", handleClickOutside);return lambda -> None { document.removeEventListener(
            "mousedown", handleClickOutside
        );}; },
        []
    );

    def handleSelect(option: any) {
        onChange(option["value"]);
        setIsOpen(False);
        setSearchTerm("");
    }

    selectedOption = options.find(
        lambda (o: any) -> bool { return o["value"] == value; }
    );

    return (
        <div className="relative" ref={wrapperRef}>
            <div
                className={"w-full bg-zinc-900 border border-zinc-700 rounded-lg flex items-center justify-between p-2.5 text-sm cursor-pointer transition-colors " + (
                    "opacity-50 cursor-not-allowed"
                    if disabled
                    else "hover:border-zinc-600"
                )}
                onClick={lambda -> None { if not disabled {
                    setIsOpen(not isOpen);
                }}}
            >
                <span
                    className={"block truncate " + (
                        "text-zinc-500" if not selectedOption else "text-zinc-300"
                    )}
                >
                    {selectedOption["label"] if selectedOption else placeholder}
                </span>
                <ChevronDownIcon className="w-4 h-4 text-zinc-500"/>
            </div>
            {createPortal(
                <div
                    ref={portalRef}
                    style={{
                        "position": "fixed",
                        "top": coords.top,
                        "left": coords.left,
                        "width": coords.width,
                        "zIndex": 9999
                    }}
                    className="mt-1 bg-zinc-900 border border-zinc-700 rounded-lg shadow-2xl max-h-[400px] overflow-hidden flex flex-col"
                >
                    <div className="p-2 border-b border-zinc-800">
                        <div
                            className="flex items-center px-2 bg-zinc-950 rounded border border-zinc-800"
                        >
                            <SearchIcon className="w-4 h-4 text-zinc-500 mr-2"/>
                            <input
                                type="text"
                                className="w-full py-1.5 bg-transparent border-none focus:outline-none text-sm text-zinc-300 placeholder-zinc-600"
                                placeholder="Search..."
                                value={searchTerm}
                                onChange={lambda (e: any) -> None { setSearchTerm(
                                    e.target.value
                                );}}
                                autoFocus={True}
                                onClick={lambda (e: any) -> None { e.stopPropagation();}}
                            />
                        </div>
                    </div>
                    <div className="overflow-y-auto flex-1 custom-scrollbar">
                        {(
                            filteredOptions.map(
                                lambda (option: any) -> Any { isSelected = option[
                                    "value"
                                ] == value;return (
                                    <div
                                        key={option["value"]}
                                        className={"px-3 py-2 text-sm cursor-pointer flex items-center justify-between " + (
                                            "bg-indigo-600/20 text-indigo-300"
                                            if isSelected
                                            else "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                                        )}
                                        onClick={lambda (e: any) -> None { e.stopPropagation();handleSelect(
                                            option
                                        );}}
                                    >
                                        <span className="truncate">
                                            {option["label"]}
                                        </span>
                                        {<CheckIcon className="w-4 h-4"/>
                                        if isSelected
                                        else null}
                                    </div>
                                ); }
                            )
                        )
                        if filteredOptions.length > 0
                        else <div
                            className="px-3 py-4 text-center text-xs text-zinc-600"
                        >
                            No results found
                        </div>}
                    </div>
                </div>,
                document.body
            ) if isOpen else null}
        </div>
    );
}
