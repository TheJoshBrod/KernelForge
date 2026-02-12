"""Project Page Component"""

import from react { useEffect }
import from "@jac/runtime" { jacSpawn }
import from "react-router-dom" { useNavigate, useParams }
import from recharts {
    PieChart,
    Pie,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
}
import from "backdrops/GameOfLifeBackground" { GameOfLifeBackground }
import from .GlowButton { GlowButton }
import from .Mascot { Mascot }

def:pub Project -> any {
    navigate = useNavigate();
    params = useParams();
    raw_name = params.projectName;
    name = raw_name or "";
    COLORS = ['#86e6a9', '#5bbbd3', '#b7f0d0', '#d8f3ff', '#9ae9cc', '#78b5de'];

    has ops: list = [],
        coverage: dict = {},
        loading: bool = True,
        opsLoadedOnce: bool = False,
        status: dict = {},
        benchResults: list = [],
        benchLoading: bool = False,
        exportPath: str = "",
        downloading: bool = False,
        exportSelectedOps: list = [],
        actionSelectedOps: list = [],
        actionSearchQuery: str = "",
        panelOpen: bool = False,
        panelMode: str = "",
        advOptimize: bool = True,
        advBenchmark: bool = True,
        advIterations: int = 3,
        advPatience: int = 2,
        advMinImprove: float = 0.0,
        advSpeedupPct: float = 5.0,
        advGenAttempts: int = 4,
        advImage: str = "cgins-codex:latest",
        targetDevice: str = "cuda",
        hardwareInfo: dict = {},
        hardwareLoading: bool = False,
        statusLoadedOnce: bool = False,
        autoBenchStarted: bool = False,
        asDoctorInfo: dict = {},
        asReport: dict = {},
        asModelPath: str = "",
        asPromptPath: str = "",
        asProfile: str = "both",
        asDoctorLoading: bool = False,
        asOptimizing: bool = False,
        asExportPath: str = "";

    def format_op(raw: str) -> str {
        raw_text = raw or "";
        prefix = "torch_nn_functional_";
        if raw_text.startswith(prefix) {
            return "torch.nn.functional." + raw_text.replace(prefix, "");
        }
        return raw_text.replace("_", ".");
    }

    def _job_is_active(job: dict) -> bool {
        if not job or not job["status"] {
            return False;
        }
        state = String(job["status"]).toLowerCase();
        return state == "queued" or state == "running" or state == "paused";
    }

    def _coverage_generated_ops -> list {
        if coverage and "generated" in coverage and coverage["generated"] {
            return coverage["generated"];
        }
        return [];
    }

    def _coverage_missing_ops -> list {
        if coverage and "missing" in coverage and coverage["missing"] {
            return coverage["missing"];
        }
        return [];
    }

    def _names_from_ops(items: list) -> list {
        out = [];
        for item in items {
            if item and "name" in item and item["name"] {
                out = out.concat([item["name"]]);
            }
        }
        return out;
    }

    def _contains_op(items: list, op_name: str) -> bool {
        for item in items {
            if item and "name" in item and item["name"] == op_name {
                return True;
            }
        }
        return False;
    }

    def _cache_key() -> str {
        return f"cgins_project_view::{name}";
    }

    def _hardware_cache_key() -> str {
        return "cgins_hardware_detect_v1";
    }

    def loadCachedView -> bool {
        if not name {
            return False;
        }
        raw = window.sessionStorage.getItem(_cache_key());
        if not raw {
            return False;
        }
        data = JSON.parse(raw);
        if data and "ops" in data and data["ops"] {
            ops = data["ops"];
            opsLoadedOnce = True;
            loading = False;
        }
        if data and "coverage" in data and data["coverage"] {
            coverage = data["coverage"];
        }
        if data and "status" in data and data["status"] {
            status = data["status"];
            statusLoadedOnce = True;
        }
        if data and "benchResults" in data and data["benchResults"] {
            benchResults = data["benchResults"];
            benchLoading = False;
        }
        if data and "hardwareInfo" in data and data["hardwareInfo"] {
            hardwareInfo = data["hardwareInfo"];
        }
        if data and "targetDevice" in data and data["targetDevice"] {
            targetDevice = data["targetDevice"];
        }
        if data and "asReport" in data and data["asReport"] {
            asReport = data["asReport"];
        }
        if data and "asModelPath" in data and data["asModelPath"] {
            asModelPath = data["asModelPath"];
        }
        if data and "asPromptPath" in data and data["asPromptPath"] {
            asPromptPath = data["asPromptPath"];
        }
        if data and "asProfile" in data and data["asProfile"] {
            asProfile = data["asProfile"];
        }
        if data and "asDoctorInfo" in data and data["asDoctorInfo"] {
            asDoctorInfo = data["asDoctorInfo"];
        }
        if data and "status" in data and data["status"] and "benchmark" in data["status"] {
            autoBenchStarted = True;
        }
        return True;
    }

    def cacheProjectView -> None {
        if not name {
            return;
        }
        payload = {
            "ops": ops,
            "coverage": coverage,
            "status": status,
            "benchResults": benchResults,
            "hardwareInfo": hardwareInfo,
            "targetDevice": targetDevice,
            "asReport": asReport,
            "asModelPath": asModelPath,
            "asPromptPath": asPromptPath,
            "asProfile": asProfile,
            "asDoctorInfo": asDoctorInfo
        };
        window.sessionStorage.setItem(_cache_key(), JSON.stringify(payload));
    }

    def loadCachedHardware -> bool {
        raw = window.localStorage.getItem(_hardware_cache_key());
        if not raw {
            return False;
        }
        data = JSON.parse(raw);
        if data and "hardwareInfo" in data and data["hardwareInfo"] {
            hardwareInfo = data["hardwareInfo"];
        }
        if data and "targetDevice" in data and data["targetDevice"] {
            targetDevice = data["targetDevice"];
        } elif hardwareInfo
        and "mps_available" in hardwareInfo
        and hardwareInfo["mps_available"] {
            targetDevice = "mps";
        } elif hardwareInfo and "cuda_available" in hardwareInfo {
            targetDevice = "cuda" if hardwareInfo["cuda_available"] else "cpu";
        }
        return True;
    }

    def cacheHardwareInfo -> None {
        if not hardwareInfo {
            return;
        }
        payload = {"hardwareInfo": hardwareInfo, "targetDevice": targetDevice};
        window.localStorage.setItem(_hardware_cache_key(), JSON.stringify(payload));
    }

    async def refreshOps(showLoading: bool = True) -> None {
        if showLoading and not opsLoadedOnce {
            loading = True;
        }
        try {
            if name {
                resp = await jacSpawn("GetProjectOps", "", {"projectName": name});
                data = resp.reports[0]
                if resp and resp.reports and resp.reports.length > 0
                else None;
                if data and data["ops"] {
                    ops = data["ops"];
                } else {
                    ops = [];
                }
                opsLoadedOnce = True;
            } else {
                ops = [];
                opsLoadedOnce = True;
            }
        } except (error) {
            console.error("Failed to fetch ops:", error);
            if not opsLoadedOnce {
                ops = [];
            }
        } finally {
            if showLoading {
                loading = False;
            }
        }
    }

    async def refreshCoverage -> None {
        try {
            if name {
                resp = await jacSpawn(
                    "GetProjectCoverage", "", {"projectName": name}
                );
                data = resp.reports[0]
                if resp and resp.reports and resp.reports.length > 0
                else {};
                coverage = data or {};
            } else {
                coverage = {};
            }
        } except (error) {
            console.error("Failed to fetch coverage:", error);
            coverage = {};
        }
    }

    async def refreshStatus -> None {
        try {
            if name {
                resp = await jacSpawn("GetProjectStatus", "", {"projectName": name});
                data = resp.reports[0]
                if resp and resp.reports and resp.reports.length > 0
                else {};
                status = data or {};
                statusLoadedOnce = True;
            } else {
                status = {};
                statusLoadedOnce = True;
            }
        } except (error) {
            console.error("Failed to fetch status:", error);
            status = {};
            statusLoadedOnce = True;
        }
    }

    async def refreshBench(showLoading: bool = True) -> None {
        if showLoading {
            benchLoading = True;
        }
        try {
            if name {
                resp = await jacSpawn(
                    "GetProjectBenchmarks", "", {"projectName": name}
                );
                data = resp.reports[0]
                if resp and resp.reports and resp.reports.length > 0
                else {};
                if data and data["results"] {
                    benchResults = data["results"];
                    if exportSelectedOps.length == 0 and data["winners"] {
                        exportSelectedOps = data["winners"];
                    }
                } else {
                    benchResults = [];
                }
            } else {
                benchResults = [];
            }
        } except (error) {
            console.error("Failed to fetch benchmarks:", error);
            benchResults = [];
        } finally {
            if showLoading {
                benchLoading = False;
            }
        }
    }

    async def refreshAll -> None {
        await refreshOps();
        await refreshCoverage();
        await refreshStatus();
        await refreshBench();
        await refreshAppleSiliconReport();
    }

    async def detectHardware -> None {
        hardwareLoading = True;
        try {
            resp = await jacSpawn("DetectHardware", "", {});
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else {};
            hardwareInfo = data or {};
            if hardwareInfo
            and "mps_available" in hardwareInfo
            and hardwareInfo["mps_available"] {
                targetDevice = "mps";
            } elif hardwareInfo and "cuda_available" in hardwareInfo {
                if not hardwareInfo["cuda_available"] {
                    targetDevice = "cpu";
                } else {
                    targetDevice = "cuda";
                }
            }
            cacheHardwareInfo();
        } except (error) {
            console.error("Failed to detect hardware:", error);
            hardwareInfo = {};
        } finally {
            hardwareLoading = False;
        }
    }

    async def setJobControl(jobKey: str, action: str) -> None {
        if name {
            await jacSpawn(
                "SetJobControl",
                "",
                {"projectName": name, "jobKey": jobKey, "action": action}
            );
            await refreshStatus();
        }
    }

    async def handleProfile -> None {
        if name {
            await jacSpawn("StartProfile", "", {"projectName": name});
            autoBenchStarted = False;
            await refreshStatus();
            await refreshOps(False);
            await refreshCoverage();
        }
    }

    async def handleBenchmark -> None {
        if name {
            autoBenchStarted = True;
            await jacSpawn("StartBenchmark", "", {"projectName": name});
            await refreshStatus();
            await refreshBench();
        }
    }

    async def refreshAppleSiliconReport -> None {
        if not name {
            asReport = {};
            return;
        }
        try {
            resp = await jacSpawn(
                "GetAppleSiliconReport",
                "",
                {"projectName": name}
            );
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else {};
            if data and "success" in data and data["success"] and "report" in data {
                asReport = data["report"] or {};
            } else {
                asReport = {};
            }
        } except (error) {
            console.error("Failed to fetch Apple Silicon report:", error);
            asReport = {};
        }
    }

    async def handleAppleSiliconDoctor -> None {
        asDoctorLoading = True;
        try {
            resp = await jacSpawn(
                "AppleSiliconDoctor",
                "",
                {"projectName": name}
            );
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else {};
            asDoctorInfo = data or {};
        } except (error) {
            console.error("Apple Silicon doctor failed:", error);
            asDoctorInfo = {};
        } finally {
            asDoctorLoading = False;
        }
    }

    async def startAppleSiliconOptimize(gate: str) -> None {
        if not name {
            return;
        }
        asOptimizing = True;
        try {
            await jacSpawn(
                "StartAppleSiliconOptimize",
                "",
                {
                    "projectName": name,
                    "modelPath": asModelPath,
                    "profile": asProfile,
                    "gate": gate,
                    "promptsPath": asPromptPath
                }
            );
            await refreshStatus();
        } except (error) {
            console.error("Apple Silicon optimize failed to start:", error);
        } finally {
            asOptimizing = False;
        }
    }

    async def handleAppleSiliconExportPack -> None {
        if not name {
            return;
        }
        try {
            resp = await jacSpawn(
                "AppleSiliconExportPack",
                "",
                {"projectName": name, "modelPath": asModelPath}
            );
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else {};
            if data and "path" in data and data["path"] {
                asExportPath = data["path"];
            }
        } except (error) {
            console.error("Apple Silicon export failed:", error);
        }
    }

    async def handleAppleSiliconDisablePack -> None {
        if not name {
            return;
        }
        try {
            await jacSpawn(
                "AppleSiliconDisablePack",
                "",
                {"projectName": name, "modelPath": asModelPath}
            );
            await refreshAppleSiliconReport();
        } except (error) {
            console.error("Apple Silicon disable failed:", error);
        }
    }

    async def maybeStartBenchmark -> None {
        if not name or autoBenchStarted {
            return;
        }
        if not statusLoadedOnce {
            return;
        }
        if ops.length == 0 {
            return;
        }
        profile_job = status["profile"] if status and "profile" in status else {};
        prepare_job = status["prepare"] if status and "prepare" in status else {};
        if _job_is_active(profile_job) or _job_is_active(prepare_job) {
            return;
        }

        bench_job = status["benchmark"] if status and "benchmark" in status else {};
        if bench_job and "status" in bench_job and bench_job["status"] {
            autoBenchStarted = True;
            return;
        }
        if _job_is_active(bench_job) {
            autoBenchStarted = True;
            return;
        }
        if benchResults.length > 0 {
            autoBenchStarted = True;
            return;
        }

        autoBenchStarted = True;
        await jacSpawn("StartBenchmark", "", {"projectName": name});
        await refreshStatus();
        await refreshBench(False);
    }

    async def runGenerateAdvancedWithSelection(
        selected: list,
        optimizeFlag: bool,
        benchmarkFlag: bool
    ) -> None {
        if not name {
            return;
        }
        ops_to_use = selected;
        if ops_to_use.length == 0 {
            return;
        }
        await jacSpawn(
            "StartGenerateAdvanced",
            "",
            {
                "projectName": name,
                "ops": ops_to_use,
                "optimize": optimizeFlag,
                "benchmark": benchmarkFlag,
                "iterations": advIterations,
                "patience": advPatience,
                "minImproveMs": advMinImprove,
                "minSpeedupPct": advSpeedupPct,
                "genAttempts": advGenAttempts,
                "image": advImage,
                "targetDevice": targetDevice
            }
        );
        await refreshStatus();
    }

    async def runOptimizeAdvancedWithSelection(selected: list) -> None {
        if not name {
            return;
        }
        ops_to_use = selected;
        if ops_to_use.length == 0 {
            return;
        }
        await jacSpawn(
            "StartOptimizeAdvanced",
            "",
            {
                "projectName": name,
                "ops": ops_to_use,
                "iterations": advIterations,
                "patience": advPatience,
                "minImproveMs": advMinImprove,
                "minSpeedupPct": advSpeedupPct,
                "image": advImage,
                "targetDevice": targetDevice
            }
        );
        await refreshStatus();
    }

    async def handleExport -> None {
        if name {
            export_ops = exportSelectedOps;
            if export_ops.length == 0 {
                winners = [];
                for r in benchResults {
                    if r and "winner" in r and r["winner"] {
                        winners = winners.concat([r["op"]]);
                    }
                }
                export_ops = winners;
            }
            resp = await jacSpawn(
                "ExportProject", "", {"projectName": name, "ops": export_ops}
            );
            result = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            if result and result["success"] {
                exportPath = result["path"];
            }
        }
    }

    async def handleDownloadExport -> None {
        if name {
            downloading = True;
            try {
                export_ops = exportSelectedOps;
                if export_ops.length == 0 {
                    winners = [];
                    for r in benchResults {
                        if r and "winner" in r and r["winner"] {
                            winners = winners.concat([r["op"]]);
                        }
                    }
                    export_ops = winners;
                }
                resp = await jacSpawn(
                    "DownloadExport", "", {"projectName": name, "ops": export_ops}
                );
                data = resp.reports[0]
                if resp and resp.reports and resp.reports.length > 0
                else None;
                if data and data["success"] {
                    file_name = data["name"] if data["name"] else f"{name}.cgins.zip";
                    data_uri = f"data:application/zip;base64,{data['data']}";
                    blob_resp = await fetch(data_uri);
                    blob = await blob_resp.blob();
                    url = URL.createObjectURL(blob);
                    link = document.createElement("a");
                    link.href = url;
                    link.download = file_name;
                    link.click();
                    URL.revokeObjectURL(url);
                }
            } except (error) {
                console.error("Failed to download export:", error);
            } finally {
                downloading = False;
            }
        }
    }

    def openActionPanel(mode: str) -> None {
        panelMode = mode;
        panelOpen = True;
        actionSearchQuery = "";

        initial = [];
        if mode == "optimize" {
            initial = _names_from_ops(_coverage_generated_ops());
        } elif mode == "generate" {
            initial = _names_from_ops(_coverage_missing_ops());
            if initial.length == 0 {
                initial = _names_from_ops(ops);
            }
        } else {
            initial = _names_from_ops(ops);
        }
        actionSelectedOps = initial;
    }

    def closeActionPanel -> None {
        panelOpen = False;
        panelMode = "";
        actionSearchQuery = "";
    }

    def toggleActionSelection(op_name: str) -> None {
        if actionSelectedOps.includes(op_name) {
            new_list = [];
            for op in actionSelectedOps {
                if op != op_name {
                    new_list = new_list.concat([op]);
                }
            }
            actionSelectedOps = new_list;
        } else {
            actionSelectedOps = actionSelectedOps.concat([op_name]);
        }
    }

    def selectAllActionOps(options: list) -> None {
        actionSelectedOps = _names_from_ops(options);
    }

    def clearActionSelection -> None {
        actionSelectedOps = [];
    }

    def toggleExportSelection(op_name: str) -> None {
        if exportSelectedOps.includes(op_name) {
            new_list = [];
            for op in exportSelectedOps {
                if op != op_name {
                    new_list = new_list.concat([op]);
                }
            }
            exportSelectedOps = new_list;
        } else {
            exportSelectedOps = exportSelectedOps.concat([op_name]);
        }
    }

    def chooseWinnersForExport -> None {
        winners = [];
        for r in benchResults {
            if r and "winner" in r and r["winner"] {
                winners = winners.concat([r["op"]]);
            }
        }
        exportSelectedOps = winners;
    }

    async def runPanelOptimize -> None {
        generated_options = _coverage_generated_ops();
        selected = actionSelectedOps;
        if selected.length == 0 {
            selected = _names_from_ops(generated_options);
        }
        await runOptimizeAdvancedWithSelection(selected);
        closeActionPanel();
    }

    async def runPanelGenerate -> None {
        options = ops;
        selected = actionSelectedOps;
        if selected.length == 0 {
            selected = _names_from_ops(options);
        }
        await runGenerateAdvancedWithSelection(selected, False, False);
        closeActionPanel();
    }

    async def runPanelAdvancedGenerate -> None {
        options = ops;
        selected = actionSelectedOps;
        if selected.length == 0 {
            selected = _names_from_ops(options);
        }
        await runGenerateAdvancedWithSelection(selected, advOptimize, advBenchmark);
        closeActionPanel();
    }

    async def runPanelAdvancedOptimize -> None {
        options = ops;
        selected = actionSelectedOps;
        if selected.length == 0 {
            selected = _names_from_ops(options);
        }
        await runOptimizeAdvancedWithSelection(selected);
        closeActionPanel();
    }

    profile_status = status["profile"]["status"]
    if status
    and "profile" in status
    and status["profile"]
    and "status" in status["profile"]
    else "";
    prepare_status = status["prepare"]["status"]
    if status
    and "prepare" in status
    and status["prepare"]
    and "status" in status["prepare"]
    else "";
    generate_status = status["generate"]["status"]
    if status
    and "generate" in status
    and status["generate"]
    and "status" in status["generate"]
    else "";
    generate_adv_status = status["generate_adv"]["status"]
    if status
    and "generate_adv" in status
    and status["generate_adv"]
    and "status" in status["generate_adv"]
    else "";
    optimize_status = status["optimize"]["status"]
    if status
    and "optimize" in status
    and status["optimize"]
    and "status" in status["optimize"]
    else "";
    optimize_adv_status = status["optimize_adv"]["status"]
    if status
    and "optimize_adv" in status
    and status["optimize_adv"]
    and "status" in status["optimize_adv"]
    else "";
    benchmark_status = status["benchmark"]["status"]
    if status
    and "benchmark" in status
    and status["benchmark"]
    and "status" in status["benchmark"]
    else "";
    apple_silicon_optimize_status = status["apple_silicon_optimize"]["status"]
    if status
    and "apple_silicon_optimize" in status
    and status["apple_silicon_optimize"]
    and "status" in status["apple_silicon_optimize"]
    else "";

    useEffect(
        lambda -> None {
            autoBenchStarted = False;
            restored = loadCachedView();
            has_cached_hw = loadCachedHardware();
            refreshOps(not restored);
            refreshCoverage();
            refreshStatus();
            refreshBench(not restored);
            refreshAppleSiliconReport();
            if not has_cached_hw {
                detectHardware();
            }
            if not asDoctorInfo {
                handleAppleSiliconDoctor();
            }
        },
        [name]
    );

    useEffect(
        lambda -> None { cacheProjectView(); },
        [
            name,
            ops,
            coverage,
            status,
            benchResults,
            hardwareInfo,
            targetDevice,
            asReport,
            asModelPath,
            asPromptPath,
            asProfile,
            asDoctorInfo
        ]
    );

    useEffect(
        lambda -> None { cacheHardwareInfo(); },
        [hardwareInfo, targetDevice]
    );

    useEffect(
        lambda -> None {
            timer = setInterval(lambda -> None { refreshStatus();}, 4000);
            return lambda -> None { clearInterval(timer); };
        },
        [name]
    );

    useEffect(
        lambda -> None {
            profile_active = _job_is_active(status["profile"])
            if status and "profile" in status
            else False;
            prepare_active = _job_is_active(status["prepare"])
            if status and "prepare" in status
            else False;
            generate_active = _job_is_active(status["generate"])
            if status and "generate" in status
            else False;
            generate_adv_active = _job_is_active(status["generate_adv"])
            if status and "generate_adv" in status
            else False;
            optimize_active = _job_is_active(status["optimize"])
            if status and "optimize" in status
            else False;
            optimize_adv_active = _job_is_active(status["optimize_adv"])
            if status and "optimize_adv" in status
            else False;
            bench_active = _job_is_active(status["benchmark"])
            if status and "benchmark" in status
            else False;
            as_opt_active = _job_is_active(status["apple_silicon_optimize"])
            if status and "apple_silicon_optimize" in status
            else False;

            if (
                not profile_active
                and not prepare_active
                and not generate_active
                and not generate_adv_active
                and not optimize_active
                and not optimize_adv_active
                and not bench_active
                and not as_opt_active
            ) {
                return lambda -> None {};
            }

            timer = setInterval(
                lambda -> None {
                    refreshStatus();
                    refreshOps(False);
                    refreshCoverage();
                    refreshBench(False);
                    refreshAppleSiliconReport();
                },
                1000
            );
            return lambda -> None { clearInterval(timer); };
        },
        [
            name,
            profile_status,
            prepare_status,
            generate_status,
            generate_adv_status,
            optimize_status,
            optimize_adv_status,
            benchmark_status,
            apple_silicon_optimize_status
        ]
    );

    useEffect(
        lambda -> None { maybeStartBenchmark(); },
        [
            name,
            ops.length,
            benchResults.length,
            benchmark_status,
            autoBenchStarted,
            statusLoadedOnce,
            profile_status,
            prepare_status
        ]
    );

    useEffect(
        lambda -> None {
            if name {
                refreshAppleSiliconReport();
            }
        },
        [name, apple_silicon_optimize_status]
    );

    generated_ops = _coverage_generated_ops();
    missing_ops = _coverage_missing_ops();

    generated_count = coverage["generated_count"]
    if coverage and "generated_count" in coverage
    else generated_ops.length;
    missing_count = coverage["missing_count"]
    if coverage and "missing_count" in coverage
    else missing_ops.length;
    total_count = coverage["total_count"]
    if coverage and "total_count" in coverage
    else ops.length;

    generatedPreview = [];
    generated_preview_count = 0;
    for item in generated_ops {
        if generated_preview_count < 8 {
            op_name = item["name"];
            generatedPreview = generatedPreview.concat(
                [
                    <button
                        key={"gen-" + op_name}
                        className="coverage-op"
                        onClick={lambda -> None { navigate(
                            f"/project/{name}/{op_name}"
                        );}}
                    >
                        {format_op(op_name)}
                    </button>
                ]
            );
            generated_preview_count = generated_preview_count + 1;
        }
    }

    missingPreview = [];
    missing_preview_count = 0;
    for item in missing_ops {
        if missing_preview_count < 8 {
            op_name = item["name"];
            missingPreview = missingPreview.concat(
                [
                    <button
                        key={"miss-" + op_name}
                        className="coverage-op coverage-op--missing"
                        onClick={lambda -> None { navigate(
                            f"/project/{name}/{op_name}"
                        );}}
                    >
                        {format_op(op_name)}
                    </button>
                ]
            );
            missing_preview_count = missing_preview_count + 1;
        }
    }

    pieData = [];
    hasPie = False;
    pie_count = 0;
    for op in ops {
        if pie_count < 6 {
            pieData = pieData.concat(
                [
                    {
                        "name": format_op(op["name"]),
                        "value": op["count"],
                        "fill": COLORS[pie_count]
                    }
                ]
            );
            hasPie = True;
            pie_count = pie_count + 1;
        }
    }

    benchByOp = {};
    for row in benchResults {
        if row and "op" in row and row["op"] {
            benchByOp[row["op"]] = row;
        }
    }

    usageChartData = [];
    usage_chart_count = 0;
    hasUsageChart = False;
    for op in ops {
        if usage_chart_count < 12 {
            op_name = op["name"];
            call_count = op["count"] if op and "count" in op else 0;
            bench_row = benchByOp[op_name] if op_name in benchByOp else {};
            pytorch_ms = bench_row["pytorch_ms"]
            if bench_row and "pytorch_ms" in bench_row and bench_row["pytorch_ms"]
            else 0.0;
            normalized_ms = bench_row["normalized_ms"]
            if bench_row and "normalized_ms" in bench_row and bench_row["normalized_ms"]
            else 0.0;
            label = format_op(op_name);
            if label.length > 26 {
                label = f"{label.slice(0, 23)}...";
            }
            usageChartData = usageChartData.concat(
                [
                    {
                        "label": label,
                        "full": format_op(op_name),
                        "usage_count": call_count,
                        "pytorch_ms": pytorch_ms,
                        "normalized_ms": normalized_ms
                    }
                ]
            );
            hasUsageChart = True;
            usage_chart_count = usage_chart_count + 1;
        }
    }

    statusItems = [
        {"key": "profile", "label": "Profile"},
        {"key": "prepare", "label": "Prepare"},
        {"key": "generate", "label": "Generate"},
        {"key": "generate_adv", "label": "Adv Generate"},
        {"key": "optimize", "label": "Optimize"},
        {"key": "optimize_adv", "label": "Adv Optimize"},
        {"key": "benchmark", "label": "Benchmark"},
        {"key": "apple_silicon_optimize", "label": "Apple Silicon Optimize"}
    ];

    def status_label(key: str) -> str {
        job = status[key] if status and key in status else {};
        if job and "status" in job and job["status"] {
            return job["status"];
        }
        return "idle";
    }

    statusRows = [];
    for item in statusItems {
        job = status[item["key"]] if status and item["key"] in status else {};
        progress = job["progress"] if job and "progress" in job else {};
        total = progress["total"] if progress and "total" in progress else 0;
        current = progress["current"] if progress and "current" in progress else 0;
        pct = progress["percent"] * 100 if progress and "percent" in progress else 0;
        bar_width = f"{pct}%" if pct > 2 else "8%";
        show_progress = (
            job
            and (job["status"] == "running" or job["status"] == "paused")
        );
        message = job["message"] if job and "message" in job else "";
        progress_text = f"{current}/{total}" if total > 0 else "working...";

        controls = <span></span>;
        if (
            item["key"] == "generate"
            or item["key"] == "generate_adv"
            or item["key"] == "optimize"
            or item["key"] == "optimize_adv"
            or item["key"] == "apple_silicon_optimize"
        ) {
            if job and job["status"] == "running" {
                controls = <div className="status-controls">
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "pause");}}
                        className="chip-button"
                    >
                        Pause
                    </button>
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "cancel");}}
                        className="chip-button chip-danger"
                    >
                        Cancel
                    </button>
                </div>;
            } elif job and job["status"] == "paused" {
                controls = <div className="status-controls">
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "resume");}}
                        className="chip-button"
                    >
                        Resume
                    </button>
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "cancel");}}
                        className="chip-button chip-danger"
                    >
                        Cancel
                    </button>
                </div>;
            }
        }

        statusRows = statusRows.concat(
            [
                <div key={item["key"]} className="status-row">
                    <div className="status-row-header">
                        <span className="status-label">
                            {item["label"]}
                        </span>
                        <span className="status-pill">
                            {status_label(item["key"])}
                        </span>
                    </div>
                    {<div className="status-progress">
                        <div className="progress-track">
                            <div
                                className="progress-fill"
                                style={{"width": bar_width}}
                            ></div>
                        </div>
                        <div className="progress-meta">
                            <span>
                                {progress_text}
                            </span>
                            <span className="progress-message">
                                {message}
                            </span>
                        </div>
                        {controls}
                    </div>
                    if show_progress
                    else <span></span>}
                </div>
            ]
        );
    }

    bench_job = status["benchmark"] if status and "benchmark" in status else {};
    bench_progress = bench_job["progress"]
    if bench_job and "progress" in bench_job
    else {};
    bench_total = bench_progress["total"]
    if bench_progress and "total" in bench_progress
    else 0;
    bench_current = bench_progress["current"]
    if bench_progress and "current" in bench_progress
    else 0;
    bench_pct = bench_progress["percent"] * 100
    if bench_progress and "percent" in bench_progress
    else 0;
    bench_bar_width = f"{bench_pct}%" if bench_pct > 2 else "8%";
    bench_message = bench_job["message"]
    if bench_job and "message" in bench_job
    else "";
    show_bench_progress = (
        bench_job
        and (bench_job["status"] == "running" or bench_job["status"] == "paused")
    );

    benchRows = [];
    hasBench = False;
    for r in benchResults {
        op_name = r["op"];
        speedup = r["speedup"];
        kernel_status = r["kernel_status"] if r and "kernel_status" in r else "";
        kernel_error = r["kernel_error"] if r and "kernel_error" in r else "";
        kernel_ms = r["kernel_ms"];
        kernel_text = "n/a";
        kernel_class = "table-muted";
        if kernel_status == "ok" {
            kernel_text = f"{kernel_ms:.6f}" if kernel_ms else "0";
            kernel_class = "";
        } elif kernel_status == "missing" {
            kernel_text = "missing";
            kernel_class = "table-muted";
        } elif kernel_status == "error" {
            kernel_text = "error";
            kernel_class = "table-error";
        } elif kernel_status == "skipped" {
            kernel_text = "skipped";
            kernel_class = "table-muted";
        }
        speedup_text = (
            f"{speedup:.2f}x" if (kernel_status == "ok" and speedup) else "n/a"
        );
        is_win = r["winner"];
        source_text = r["source"] if r and r["source"] else "";
        if kernel_status == "missing" {
            source_text = "missing";
        } elif kernel_status == "error" {
            source_text = f"{source_text} (error)";
        }
        checked = exportSelectedOps.includes(op_name);

        benchRows = benchRows.concat(
            [
                <tr key={op_name} className="table-row">
                    <td className="table-cell mono">
                        {format_op(op_name)}
                    </td>
                    <td className="table-cell">
                        {r["pytorch_ms"]}
                    </td>
                    <td className={"table-cell " + kernel_class}>
                        <span title={kernel_error}>
                            {kernel_text}
                        </span>
                    </td>
                    <td className="table-cell">
                        <span
                            className={("status-success" if is_win else "table-muted")}
                        >
                            {speedup_text}
                        </span>
                    </td>
                    <td className="table-cell table-muted">
                        {source_text}
                    </td>
                    <td className="table-cell text-center">
                        <input
                            type="checkbox"
                            checked={checked}
                            onChange={lambda e: any  -> None { toggleExportSelection(
                                op_name
                            );}}
                        />
                    </td>
                </tr>
            ]
        );
        hasBench = True;
    }

    hardwareStatus = "";
    if hardwareInfo
    and "mps_available" in hardwareInfo
    and hardwareInfo["mps_available"] {
        hardwareStatus = "MPS: Apple Silicon";
    } elif hardwareInfo and "cuda_available" in hardwareInfo {
        if hardwareInfo["cuda_available"] {
            gpu_name = hardwareInfo["device_name"]
            if "device_name" in hardwareInfo
            else "";
            count = hardwareInfo["device_count"]
            if "device_count" in hardwareInfo
            else 0;
            suffix = f" x{count}" if count and count > 1 else "";
            hardwareStatus = f"CUDA: {gpu_name}{suffix}";
        } else {
            hardwareStatus = "CUDA not detected";
        }
    }

    profileRuntimeStatus = "";
    profileJob = status["profile"] if status and "profile" in status else {};
    profileRuntime = profileJob["runtime"]
    if profileJob and "runtime" in profileJob
    else {};
    runtimeDevice = profileRuntime["device"]
    if profileRuntime and "device" in profileRuntime
    else "";
    if runtimeDevice == "cuda" {
        runtimeName = profileRuntime["gpu_name"]
        if "gpu_name" in profileRuntime
        else "Unknown CUDA GPU";
        runtimeIndex = profileRuntime["gpu_index"]
        if "gpu_index" in profileRuntime
        else None;
        hasRuntimeIndex = ("gpu_index" in profileRuntime and runtimeIndex != None);
        indexLabel = f"GPU {runtimeIndex}" if hasRuntimeIndex else "GPU";
        profileRuntimeStatus = f"Profile runtime: CUDA ({indexLabel}) - {runtimeName}";
    } elif runtimeDevice == "mps" {
        profileRuntimeStatus = "Profile runtime: Apple MPS";
    } elif runtimeDevice == "cpu" {
        profileRuntimeStatus = "Profile runtime: CPU";
    }

    asJob = status["apple_silicon_optimize"]
    if status and "apple_silicon_optimize" in status
    else {};
    asJobMessage = asJob["message"] if asJob and "message" in asJob else "";
    asJobStatus = asJob["status"] if asJob and "status" in asJob else "";
    asHasReport = asReport and "success" in asReport and asReport["success"];
    asPackId = asReport["pack_id"] if asHasReport and "pack_id" in asReport else "";
    asPassGate = asReport["pass_gate"]
    if asHasReport and "pass_gate" in asReport
    else False;
    asProfileDelta = asReport["delta"]["profiles"]
    if asHasReport and "delta" in asReport and asReport["delta"] and "profiles" in asReport["delta"]
    else {};
    asDeltaRows = [];
    if asProfileDelta {
        for key in asProfileDelta {
            row = asProfileDelta[key] if key in asProfileDelta else {};
            pre = row["prefill_uplift_pct"] if row and "prefill_uplift_pct" in row else None;
            dec = row["decode_uplift_pct"] if row and "decode_uplift_pct" in row else None;
            asDeltaRows = asDeltaRows.concat(
                [
                    <li key={"as-delta-" + key} className="table-muted">
                        {key + ": prefill="
                        + (String(pre) if pre != None else "n/a")
                        + "% decode="
                        + (String(dec) if dec != None else "n/a")
                        + "%"}
                    </li>
                ]
            );
        }
    }

    panelSelectableOps = _coverage_generated_ops()
    if panelMode == "optimize"
    else ops;

    panelTitle = "Optimize generated kernels";
    panelHint = "Only generated kernels are selectable for optimization.";
    if panelMode == "generate" {
        panelTitle = "Generate kernels";
        panelHint = "Choose any profiled ops to generate kernels for.";
    } elif panelMode == "advanced" {
        panelTitle = "Advanced kernel actions";
        panelHint = "Generate and optimize with Codex loop settings.";
    }

    panelRows = [];
    panelVisibleCount = 0;
    lowerQuery = actionSearchQuery.toLowerCase() if actionSearchQuery else "";
    generated_for_gate = _coverage_generated_ops();
    for item in panelSelectableOps {
        if not item or not item["name"] {
            continue;
        }
        op_name = item["name"];
        label = format_op(op_name);
        if lowerQuery and not label.toLowerCase().includes(lowerQuery) {
            continue;
        }
        checked = actionSelectedOps.includes(op_name);
        isGenerated = _contains_op(generated_for_gate, op_name);
        isDisabled = (panelMode == "optimize" and not isGenerated);

        panelRows = panelRows.concat(
            [
                <label
                    key={"panel-" + op_name}
                    className={
                        "drawer-op-row "
                        + ("drawer-op-row--disabled" if isDisabled else "")
                    }
                >
                    <input
                        type="checkbox"
                        checked={checked}
                        disabled={isDisabled}
                        onChange={lambda e: any  -> None { toggleActionSelection(op_name);}}
                    />
                    <span className="mono drawer-op-name">
                        {label}
                    </span>
                    <span className="drawer-op-meta">
                        {item["count"] if item and "count" in item else 0} calls
                    </span>
                </label>
            ]
        );
        panelVisibleCount = panelVisibleCount + 1;
    }

    return
        <div className="app-shell">
            <GameOfLifeBackground />
            <div className="app-backdrop"></div>
            <div className="project-page app-content-layer">
                <header className="project-topbar">
                    <div className="brand">
                        <Mascot variant="logo" ambient={True} />
                        <div className="brand-meta">
                            <span>
                                Project
                            </span>
                            <span className="brand-tag">
                                {name}
                            </span>
                        </div>
                    </div>
                    <div className="sidebar-actions">
                        <button
                            onClick={lambda -> None { navigate("/");}}
                            className="ghost-button"
                        >
                            Home
                        </button>
                        <button
                            onClick={lambda -> None { navigate(
                                "/settings?returnTo=" + encodeURIComponent(
                                    "/project/" + name
                                )
                            );}}
                            className="ghost-button"
                        >
                            Settings
                        </button>
                    </div>
                </header>

                <main className="project-main">
                    <div className="page-header">
                        <div>
                            <h1 className="page-title">
                                Optimize with generated kernels
                            </h1>
                            <div className="badge-row">
                                <span className="badge">
                                    Total ops: {total_count}
                                </span>
                                <span className="badge">
                                    Generated: {generated_count}
                                </span>
                                <span className="badge">
                                    Missing: {missing_count}
                                </span>
                                <span className="badge">
                                    Benchmarks: {benchResults.length}
                                </span>
                            </div>
                        </div>
                        <Mascot variant="cta" ambient={True} />
                    </div>

                    <div className="project-summary-grid">
                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Kernel coverage
                                </span>
                                <span className="table-muted">
                                    loaded immediately
                                </span>
                            </div>
                            <div className="panel-body">
                                <div className="coverage-kpi-grid">
                                    <div className="coverage-kpi">
                                        <span className="coverage-kpi-label">
                                            generated
                                        </span>
                                        <span className="coverage-kpi-value">
                                            {generated_count}
                                        </span>
                                    </div>
                                    <div className="coverage-kpi coverage-kpi--warn">
                                        <span className="coverage-kpi-label">
                                            missing
                                        </span>
                                        <span className="coverage-kpi-value">
                                            {missing_count}
                                        </span>
                                    </div>
                                </div>
                                <div className="coverage-columns">
                                    <div className="coverage-column">
                                        <span className="coverage-column-title">
                                            Generated kernels
                                        </span>
                                        <div className="coverage-list">
                                            {generatedPreview
                                            if generatedPreview.length > 0
                                            else <p className="table-muted">
                                                No generated kernels yet.
                                            </p>}
                                        </div>
                                    </div>
                                    <div className="coverage-column">
                                        <span className="coverage-column-title">
                                            Missing kernels
                                        </span>
                                        <div className="coverage-list">
                                            {missingPreview
                                            if missingPreview.length > 0
                                            else <p className="table-muted">
                                                No missing ops.
                                            </p>}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Runtime hardware
                                </span>
                                <button
                                    onClick={lambda -> None { detectHardware();}}
                                    className="chip-button"
                                >
                                    {"Detecting..."
                                    if hardwareLoading
                                    else "Detect"}
                                </button>
                            </div>
                            <div className="panel-body">
                                <p className="table-muted">
                                    {hardwareStatus or "Hardware not detected yet."}
                                </p>
                                {<p className="table-muted">
                                    {profileRuntimeStatus}
                                </p>
                                if profileRuntimeStatus
                                else <span></span>}
                                <label className="input-label">
                                    Target device
                                </label>
                                <div className="hardware-target-grid">
                                    {<label className="checkbox-row">
                                        <input
                                            type="radio"
                                            checked={targetDevice == "mps"}
                                            onChange={lambda e: any  -> None { targetDevice = "mps";}}
                                        />
                                        <span>
                                            Apple MPS
                                        </span>
                                    </label>
                                    if hardwareInfo
                                    and "mps_available" in hardwareInfo
                                    and hardwareInfo["mps_available"]
                                    else <span></span>}
                                    <label className="checkbox-row">
                                        <input
                                            type="radio"
                                            checked={targetDevice == "cuda"}
                                            onChange={lambda e: any  -> None { targetDevice = "cuda";}}
                                        />
                                        <span>
                                            CUDA
                                        </span>
                                    </label>
                                    <label className="checkbox-row">
                                        <input
                                            type="radio"
                                            checked={targetDevice == "cpu"}
                                            onChange={lambda e: any  -> None { targetDevice = "cpu";}}
                                        />
                                        <span>
                                            CPU
                                        </span>
                                    </label>
                                </div>
                                <button
                                    onClick={lambda -> None { handleProfile();}}
                                    className="ghost-button"
                                >
                                    Profile Model
                                </button>
                            </div>
                        </div>

                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Apple Silicon (llama.cpp)
                                </span>
                                <button
                                    onClick={lambda -> None { handleAppleSiliconDoctor();}}
                                    className="chip-button"
                                >
                                    {"Checking..."
                                    if asDoctorLoading
                                    else "Doctor"}
                                </button>
                            </div>
                            <div className="panel-body">
                                <label className="input-label">
                                    GGUF model path (optional)
                                </label>
                                <input
                                    type="text"
                                    value={asModelPath}
                                    onChange={lambda e: any  -> None { asModelPath = e.target.value;}}
                                    placeholder="leave blank to use default tiny Qwen"
                                    className="input input--mono"
                                />
                                <label className="input-label">
                                    Prompt file (optional)
                                </label>
                                <input
                                    type="text"
                                    value={asPromptPath}
                                    onChange={lambda e: any  -> None { asPromptPath = e.target.value;}}
                                    placeholder="newline-delimited prompts"
                                    className="input input--mono"
                                />
                                <label className="input-label">
                                    Profile scope
                                </label>
                                <div className="hardware-target-grid">
                                    <label className="checkbox-row">
                                        <input
                                            type="radio"
                                            checked={asProfile == "chat"}
                                            onChange={lambda e: any  -> None { asProfile = "chat";}}
                                        />
                                        <span>
                                            Chat
                                        </span>
                                    </label>
                                    <label className="checkbox-row">
                                        <input
                                            type="radio"
                                            checked={asProfile == "long"}
                                            onChange={lambda e: any  -> None { asProfile = "long";}}
                                        />
                                        <span>
                                            Long
                                        </span>
                                    </label>
                                    <label className="checkbox-row">
                                        <input
                                            type="radio"
                                            checked={asProfile == "both"}
                                            onChange={lambda e: any  -> None { asProfile = "both";}}
                                        />
                                        <span>
                                            Both
                                        </span>
                                    </label>
                                </div>
                                <div className="button-row">
                                    <button
                                        onClick={lambda -> None { startAppleSiliconOptimize("quick");}}
                                        className="ghost-button"
                                        disabled={asOptimizing}
                                    >
                                        Quick optimize
                                    </button>
                                    <button
                                        onClick={lambda -> None { startAppleSiliconOptimize("full");}}
                                        className="ghost-button"
                                        disabled={asOptimizing}
                                    >
                                        Full optimize
                                    </button>
                                </div>
                                <div className="button-row">
                                    <button
                                        onClick={lambda -> None { handleAppleSiliconExportPack();}}
                                        className="ghost-button"
                                    >
                                        Export .cginspack
                                    </button>
                                    <button
                                        onClick={lambda -> None { handleAppleSiliconDisablePack();}}
                                        className="ghost-button"
                                    >
                                        Disable active pack
                                    </button>
                                </div>
                                {<p className="table-muted">
                                    {"Doctor commit check: "
                                    + ("ok" if asDoctorInfo and "llamacpp_commit_ok" in asDoctorInfo and asDoctorInfo["llamacpp_commit_ok"] else "not ready")}
                                </p>
                                if asDoctorInfo
                                else <span></span>}
                                {<p className="table-muted">
                                    {"Status: " + asJobStatus + " - " + asJobMessage}
                                </p>
                                if asJobStatus
                                else <span></span>}
                                {<p className="table-muted break-all">
                                    {"Pack: " + asPackId}
                                </p>
                                if asPackId
                                else <span></span>}
                                {<p className={("status-success" if asPassGate else "table-muted")}>
                                    {"Pass gate: " + ("yes" if asPassGate else "no")}
                                </p>
                                if asHasReport
                                else <span></span>}
                                {<ul className="status-list">
                                    {asDeltaRows}
                                </ul>
                                if asDeltaRows.length > 0
                                else <span></span>}
                                {<p className="table-muted break-all">
                                    {asExportPath}
                                </p>
                                if asExportPath
                                else <span></span>}
                            </div>
                        </div>
                    </div>

                    <div className="panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Actions
                            </span>
                            <span className="table-muted">
                                pop-out selectors keep the screen clean
                            </span>
                        </div>
                        <div className="action-button-grid">
                            <GlowButton
                                label="Optimize"
                                onClick={lambda -> None { openActionPanel("optimize");}}
                                variant="primary"
                            />
                            <button
                                onClick={lambda -> None { openActionPanel("generate");}}
                                className="ghost-button"
                            >
                                Generate
                            </button>
                            <button
                                onClick={lambda -> None { openActionPanel("advanced");}}
                                className="ghost-button"
                            >
                                Advanced
                            </button>
                            <button
                                onClick={lambda -> None { handleBenchmark();}}
                                className="ghost-button"
                            >
                                Benchmark
                            </button>
                            <button
                                onClick={lambda -> None { handleExport();}}
                                className="ghost-button"
                            >
                                Export .cgins
                            </button>
                            <button
                                onClick={lambda -> None { handleDownloadExport();}}
                                disabled={downloading}
                                className="ghost-button"
                            >
                                {"Downloading..."
                                if downloading
                                else "Download .cgins"}
                            </button>
                        </div>
                        {<p className="table-muted break-all">
                            {exportPath}
                        </p>
                        if exportPath
                        else <span></span>}
                    </div>

                    <div className="panel-grid">
                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Operator fraction
                                </span>
                                <span className="table-muted">
                                    top operators by usage
                                </span>
                            </div>
                            <div className="panel-body">
                                {<div className="skeleton skeleton-block"></div>
                                if loading
                                else <p className="table-muted">
                                    No profiling data yet.
                                </p>
                                if not hasPie
                                else <div className="chart-wrap">
                                    <ResponsiveContainer width="100%" height={220}>
                                        <PieChart>
                                            <Pie
                                                data={pieData}
                                                innerRadius={55}
                                                outerRadius={82}
                                                paddingAngle={3}
                                                dataKey="value"
                                            />
                                            <Tooltip
                                                cursor={{
                                                    "fill": 'rgba(15, 23, 32, 0.6)'
                                                }}
                                                contentStyle={{
                                                    "backgroundColor": '#0b1218',
                                                    "borderColor": '#1f2a36',
                                                    "borderRadius": '10px'
                                                }}
                                                itemStyle={{"color": '#e6f2ff'}}
                                                labelStyle={{"color": '#9fb1c7'}}
                                            />
                                            <Legend
                                                layout="vertical"
                                                verticalAlign="middle"
                                                align="right"
                                                wrapperStyle={{"color": '#dbe7f3'}}
                                            />
                                        </PieChart>
                                    </ResponsiveContainer>
                                </div>}
                            </div>
                        </div>

                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Op usage vs normalized PyTorch time
                                </span>
                                <span className="table-muted">
                                    normalized as pytorch_ms / op_count
                                </span>
                            </div>
                            <div className="panel-body">
                                {<div className="skeleton skeleton-block"></div>
                                if loading
                                else <p className="table-muted">
                                    No ops profiled yet.
                                </p>
                                if not hasUsageChart
                                else <div className="chart-wrap chart-wrap--tall">
                                    <ResponsiveContainer width="100%" height={300}>
                                        <BarChart data={usageChartData}>
                                            <CartesianGrid
                                                strokeDasharray="3 3"
                                                stroke="rgba(148, 163, 184, 0.18)"
                                                vertical={False}
                                            />
                                            <XAxis
                                                dataKey="label"
                                                stroke="#8fa1b5"
                                                tick={{
                                                    "fill": '#8fa1b5',
                                                    "fontSize": 11
                                                }}
                                                axisLine={False}
                                                tickLine={False}
                                                interval={0}
                                                angle={-22}
                                                textAnchor="end"
                                                height={90}
                                            />
                                            <YAxis
                                                yAxisId="left"
                                                stroke="#8fa1b5"
                                                tick={{
                                                    "fill": '#8fa1b5',
                                                    "fontSize": 11
                                                }}
                                                axisLine={False}
                                                tickLine={False}
                                            />
                                            <YAxis
                                                yAxisId="right"
                                                orientation="right"
                                                stroke="#7db1d3"
                                                tick={{
                                                    "fill": '#7db1d3',
                                                    "fontSize": 11
                                                }}
                                                axisLine={False}
                                                tickLine={False}
                                            />
                                            <Tooltip
                                                cursor={{
                                                    "fill": 'rgba(15, 23, 32, 0.6)'
                                                }}
                                                contentStyle={{
                                                    "backgroundColor": '#0b1218',
                                                    "borderColor": '#1f2a36',
                                                    "borderRadius": '10px'
                                                }}
                                                itemStyle={{"color": '#e6f2ff'}}
                                                labelStyle={{"color": '#9fb1c7'}}
                                            />
                                            <Bar
                                                yAxisId="left"
                                                dataKey="usage_count"
                                                name="calls"
                                                fill="#86e6a9"
                                                radius={[5, 5, 0, 0]}
                                            />
                                            <Bar
                                                yAxisId="right"
                                                dataKey="normalized_ms"
                                                name="PyTorch ms/call"
                                                fill="#5bbbd3"
                                                radius={[5, 5, 0, 0]}
                                            />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>}
                                {<p className="table-muted">
                                    Benchmarking is running in background to populate time bars.
                                </p>
                                if show_bench_progress and benchResults.length == 0
                                else <span></span>}
                            </div>
                        </div>
                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Pipeline status
                                </span>
                                <button
                                    onClick={lambda -> None {
                                        refreshStatus();
                                        refreshCoverage();
                                        refreshOps(False);
                                    }}
                                    className="chip-button"
                                >
                                    Refresh
                                </button>
                            </div>
                            <div className="panel-body status-list">
                                {statusRows}
                            </div>
                        </div>
                    </div>

                    <div className="panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Benchmark results
                            </span>
                            <div className="button-row">
                                <button
                                    onClick={lambda -> None { refreshBench();}}
                                    className="chip-button"
                                >
                                    Refresh
                                </button>
                                <button
                                    onClick={lambda -> None { chooseWinnersForExport();}}
                                    className="chip-button"
                                >
                                    Select winners
                                </button>
                            </div>
                        </div>
                        <div className="panel-body">
                            {<div className="skeleton skeleton-block"></div>
                            if benchLoading
                            else <p className="table-muted">
                                Benchmark is auto-started after profiling ops are available.
                            </p>
                            if not hasBench
                            else <table className="table">
                                <thead className="table-head">
                                    <tr>
                                        <th className="table-cell">
                                            Op
                                        </th>
                                        <th className="table-cell">
                                            PyTorch (ms)
                                        </th>
                                        <th className="table-cell">
                                            Kernel (ms)
                                        </th>
                                        <th className="table-cell">
                                            Speedup
                                        </th>
                                        <th className="table-cell">
                                            Source
                                        </th>
                                        <th className="table-cell text-center">
                                            Export
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {benchRows}
                                </tbody>
                            </table>}
                            {<div className="status-progress">
                                <div className="progress-track">
                                    <div
                                        className="progress-fill"
                                        style={{"width": bench_bar_width}}
                                    ></div>
                                </div>
                                <div className="progress-meta">
                                    <span>
                                        {f"{bench_current}/{bench_total}"
                                        if bench_total > 0
                                        else "working..."}
                                    </span>
                                    <span className="progress-message">
                                        {bench_message}
                                    </span>
                                </div>
                            </div>
                            if show_bench_progress
                            else <span></span>}
                        </div>
                    </div>
                </main>

                {<div
                    className="action-drawer-backdrop"
                    onClick={lambda -> None { closeActionPanel();}}
                >
                    <div
                        className="action-drawer"
                        onClick={lambda e: any  -> None { e.stopPropagation();}}
                    >
                        <div className="drawer-header">
                            <div>
                                <h2 className="drawer-title">
                                    {panelTitle}
                                </h2>
                                <p className="table-muted">
                                    {panelHint}
                                </p>
                            </div>
                            <button
                                className="chip-button"
                                onClick={lambda -> None { closeActionPanel();}}
                            >
                                Close
                            </button>
                        </div>

                        <div className="drawer-body">
                            <input
                                type="text"
                                value={actionSearchQuery}
                                onChange={lambda e: any  -> None { actionSearchQuery = e.target.value;}}
                                placeholder="Search ops..."
                                className="input"
                            />
                            <div className="button-row">
                                <button
                                    className="chip-button"
                                    onClick={lambda -> None { selectAllActionOps(
                                        panelSelectableOps
                                    );}}
                                >
                                    Select all
                                </button>
                                <button
                                    className="chip-button"
                                    onClick={lambda -> None { clearActionSelection();}}
                                >
                                    Clear
                                </button>
                                <span className="table-muted">
                                    {actionSelectedOps.length} selected
                                </span>
                            </div>

                            <div className="drawer-list">
                                {panelRows
                                if panelVisibleCount > 0
                                else <p className="table-muted">
                                    No operators match this filter.
                                </p>}
                            </div>

                            {<div className="drawer-advanced-grid">
                                <label className="checkbox-row">
                                    <input
                                        type="checkbox"
                                        checked={advBenchmark}
                                        onChange={lambda e: any  -> None { advBenchmark = not advBenchmark;}}
                                    />
                                    <span>
                                        Benchmark after advanced generate
                                    </span>
                                </label>
                                <label className="checkbox-row">
                                    <input
                                        type="checkbox"
                                        checked={advOptimize}
                                        onChange={lambda e: any  -> None { advOptimize = not advOptimize;}}
                                    />
                                    <span>
                                        Optimize during advanced generate
                                    </span>
                                </label>
                                <label className="input-label">
                                    Optimize iterations
                                </label>
                                <input
                                    type="number"
                                    min="1"
                                    value={advIterations}
                                    onChange={lambda e: any  -> None { advIterations = Number(
                                        e.target.value
                                    );}}
                                    className="input"
                                />
                                <label className="input-label">
                                    No-improve patience
                                </label>
                                <input
                                    type="number"
                                    min="1"
                                    value={advPatience}
                                    onChange={lambda e: any  -> None { advPatience = Number(
                                        e.target.value
                                    );}}
                                    className="input"
                                />
                                <label className="input-label">
                                    Min improve (ms)
                                </label>
                                <input
                                    type="number"
                                    min="0"
                                    step="0.001"
                                    value={advMinImprove}
                                    onChange={lambda e: any  -> None { advMinImprove = Number(
                                        e.target.value
                                    );}}
                                    className="input"
                                />
                                <label className="input-label">
                                    Target speedup (%)
                                </label>
                                <input
                                    type="number"
                                    min="0"
                                    step="0.1"
                                    value={advSpeedupPct}
                                    onChange={lambda e: any  -> None { advSpeedupPct = Number(
                                        e.target.value
                                    );}}
                                    className="input"
                                />
                                <label className="input-label">
                                    Generate attempts
                                </label>
                                <input
                                    type="number"
                                    min="1"
                                    value={advGenAttempts}
                                    onChange={lambda e: any  -> None { advGenAttempts = Number(
                                        e.target.value
                                    );}}
                                    className="input"
                                />
                                <label className="input-label">
                                    Docker image
                                </label>
                                <input
                                    type="text"
                                    value={advImage}
                                    onChange={lambda e: any  -> None { advImage = e.target.value;}}
                                    className="input input--mono"
                                />
                            </div>
                            if panelMode == "advanced"
                            else <span></span>}
                        </div>

                        <div className="drawer-footer">
                            {<GlowButton
                                label="Run Optimize"
                                onClick={lambda -> None { runPanelOptimize();}}
                                variant="primary"
                            />
                            if panelMode == "optimize"
                            else <span></span>}

                            {<GlowButton
                                label="Run Generate"
                                onClick={lambda -> None { runPanelGenerate();}}
                                variant="primary"
                            />
                            if panelMode == "generate"
                            else <span></span>}

                            {<div className="button-row">
                                <GlowButton
                                    label="Adv Generate"
                                    onClick={lambda -> None { runPanelAdvancedGenerate();}}
                                    variant="primary"
                                />
                                <button
                                    className="ghost-button"
                                    onClick={lambda -> None { runPanelAdvancedOptimize();}}
                                >
                                    Adv Optimize
                                </button>
                            </div>
                            if panelMode == "advanced"
                            else <span></span>}
                        </div>
                    </div>
                </div>
                if panelOpen
                else <span></span>}
            </div>
        </div>;
}
