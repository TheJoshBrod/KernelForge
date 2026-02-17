"""New Kernel Component"""

import from react { useState, useRef, useEffect }
import from "../../../components/ui/Icons.cl.jac" {
    CpuIcon
}
import from "./menus/UploadModal.cl.jac" { UploadModal }
import from "./menus/UploadTrigger.cl.jac" { UploadTrigger }
import from "./menus/GenerateMenu.cl.jac" { GenerateMenu }
import from "./menus/OptimizeMenu.cl.jac" { OptimizeMenu }

def:pub NewKernel {
    has projectName: str;

    is_open_state = useState(False);
    isOpen = is_open_state[0];
    setIsOpen = is_open_state[1];

    upload_modal_open_state = useState(False);
    uploadModalOpen = upload_modal_open_state[0];
    setUploadModalOpen = upload_modal_open_state[1];

    menu_ref = useRef(None);

    # Close menu on outside click
    useEffect(lambda -> None {
        def handle_click_outside(event: any) {
            if menu_ref.current and not menu_ref.current.contains(event.target) {
                # console.log("NewKernel: Click outside detected, closing menu.");
                setIsOpen(False);
            }
        }
        document.addEventListener("mousedown", handle_click_outside);
        return lambda -> None {
            document.removeEventListener("mousedown", handle_click_outside);
        };
    }, []);

    on_complete = lambda -> None { setIsOpen(False); };

    handle_upload_click = lambda -> None {
        console.log("NewKernel: UploadTrigger clicked, opening modal");
        setUploadModalOpen(True);
        setIsOpen(False);
    };

    return 
        <div className="relative" ref={menu_ref}>
            <button 
                className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3 px-4 rounded-lg transition-all shadow-lg hover:shadow-indigo-500/20 flex items-center justify-center gap-2"
                onClick={lambda -> None { setIsOpen(not isOpen); }}
            >
                <CpuIcon className="w-5 h-5" />
                Create New Kernel Node
            </button>
            <div className="text-center mt-2 text-[10px] text-zinc-500">
                Create a new starting point for optimization.
            </div>

            {
                <div className="absolute bottom-full left-0 w-full mb-2 bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl overflow-hidden z-20 flex flex-col">
                    <UploadTrigger onUpload={handle_upload_click} test="HELLO_DEBUG" />
                    <GenerateMenu onComplete={on_complete} />
                    <OptimizeMenu onComplete={on_complete} />
                </div>
                if isOpen
                else <span />
            }

            {console.log("NewKernel: Rendering UploadModal, isOpen:", uploadModalOpen) if True else None}
            <UploadModal 
                isOpen={uploadModalOpen} 
                onClose={lambda -> None { 
                    console.log("NewKernel: Closing modal");
                    setUploadModalOpen(False); 
                }}
                onSuccess={lambda -> None { 
                    # Refresh logic could go here
                }}
            />
        </div>;
}
