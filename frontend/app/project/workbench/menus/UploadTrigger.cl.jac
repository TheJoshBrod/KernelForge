"""Upload Trigger Component"""

import from "../../../../components/ui/Icons.cl.jac" { UploadIcon }

def:pub UploadTrigger(props: any) {
    console.log("UploadTrigger: Render", props);
    onUpload = props.onUpload;

    def handleClick -> None {
        console.log("UploadTrigger: handleClick", onUpload);
        if onUpload { onUpload(); }
    }

    return 
        <button 
            className="w-full p-3 text-left text-sm text-zinc-300 hover:bg-zinc-800 hover:text-white flex items-center gap-3 transition-colors"
            onClick={handleClick}
        >
            <UploadIcon className="w-4 h-4 text-indigo-400" />
            <div>
                <div className="font-semibold">Upload Custom Kernel</div>
                <div className="text-[10px] text-zinc-500">Import a manually written kernel</div>
            </div>
        </button>;
}
