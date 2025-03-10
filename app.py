import marimo

__generated_with = "0.11.10"
app = marimo.App(width="medium", app_title="CUT&RUN Viewer")


@app.cell
def _(mo):
    mo.md(r"""# CUT&RUN Viewer""")
    return


@app.cell
def _():
    # Define the types of datasets which can be read in
    cirro_dataset_type_filter = [
        "custom_dataset"
    ]
    return (cirro_dataset_type_filter,)


@app.cell
def _():
    # Load the marimo library in a dedicated cell for efficiency
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # If the script is running in WASM (instead of local development mode), load micropip
    import sys
    if "pyodide" in sys.modules:
        import micropip
        running_in_wasm = True
    else:
        micropip = None
        running_in_wasm = False
    return micropip, running_in_wasm, sys


@app.cell
async def _(micropip, mo, running_in_wasm):
    with mo.status.spinner("Loading dependencies"):
        # If we are running in WASM, some dependencies need to be set up appropriately.
        # This is really just aligning the needs of the app with the default library versions
        # that come when a marimo app loads in WASM.
        if running_in_wasm:
            print("Installing via micropip")
            # Downgrade plotly to avoid the use of narwhals
            await micropip.install("plotly<6.0.0")
            await micropip.install("ssl")
            await micropip.install("anndata")
            micropip.uninstall("urllib3")
            micropip.uninstall("httpx")
            await micropip.install(["urllib3==2.3.0"])
            await micropip.install([
                "boto3==1.36.23",
                "botocore==1.36.23"
            ], verbose=True)
            await micropip.install(["cirro[pyodide]>=1.2.16"], verbose=True)

        from io import StringIO, BytesIO
        from queue import Queue
        from time import sleep
        from typing import Dict, Optional, List
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        from functools import lru_cache
        import base64
        from urllib.parse import quote_plus
        from collections import defaultdict
        from functools import lru_cache

        from cirro import DataPortalLogin, DataPortalDataset
        from cirro.services.file import FileService
        from cirro.sdk.file import DataPortalFile
        from cirro.config import list_tenants

        # A patch to the Cirro client library is applied when running in WASM
        if running_in_wasm:
            from cirro.helpers import pyodide_patch_all
            pyodide_patch_all()
    return (
        BytesIO,
        DataPortalDataset,
        DataPortalFile,
        DataPortalLogin,
        Dict,
        FileService,
        List,
        Optional,
        Queue,
        StringIO,
        base64,
        defaultdict,
        go,
        list_tenants,
        lru_cache,
        np,
        pd,
        px,
        pyodide_patch_all,
        quote_plus,
        sleep,
    )


@app.cell
def _(mo):
    # Get and set the query parameters
    query_params = mo.query_params()
    return (query_params,)


@app.cell
def _(list_tenants):
    # Get the tenants (organizations) available in Cirro
    tenants_by_name = {i["displayName"]: i for i in list_tenants()}
    tenants_by_domain = {i["domain"]: i for i in list_tenants()}


    def domain_to_name(domain):
        return tenants_by_domain.get(domain, {}).get("displayName")


    def name_to_domain(name):
        return tenants_by_name.get(name, {}).get("domain")
    return domain_to_name, name_to_domain, tenants_by_domain, tenants_by_name


@app.cell
def _(mo):
    mo.md(r"""## Load Data""")
    return


@app.cell
def _(mo):
    # Use a state element to manage the Cirro client object
    get_client, set_client = mo.state(None)
    return get_client, set_client


@app.cell
def _(domain_to_name, mo, query_params, tenants_by_name):
    # Let the user select which tenant to log in to (using displayName)
    domain_ui = mo.ui.dropdown(
        options=tenants_by_name,
        value=domain_to_name(query_params.get("domain")),
        on_change=lambda i: query_params.set("domain", i["domain"]),
        label="Load Data from Cirro",
    )
    domain_ui
    return (domain_ui,)


@app.cell
def _(DataPortalLogin, domain_ui, get_client, mo):
    # If the user is not yet logged in, and a domain is selected, then give the user instructions for logging in
    # The configuration of this cell and the two below it serve the function of:
    #   1. Showing the user the login instructions if they have selected a Cirro domain
    #   2. Removing the login instructions as soon as they have completed the login flow
    if get_client() is None and domain_ui.value is not None:
        with mo.status.spinner("Authenticating"):
            # Use device code authorization to log in to Cirro
            cirro_login = DataPortalLogin(base_url=domain_ui.value["domain"])
            cirro_login_ui = mo.md(cirro_login.auth_message_markdown)
    else:
        cirro_login = None
        cirro_login_ui = None

    mo.stop(cirro_login is None)
    cirro_login_ui
    return cirro_login, cirro_login_ui


@app.cell
def _(cirro_login, set_client):
    # Once the user logs in, set the state for the client object
    set_client(cirro_login.await_completion())
    return


@app.cell
def _(get_client, mo):
    # Get the Cirro client object (but only take action if the user selected Cirro as the input)
    client = get_client()
    mo.stop(client is None)
    return (client,)


@app.cell
def _():
    # Helper functions for dealing with lists of objects that may be accessed by id or name
    def id_to_name(obj_list: list, id: str) -> str:
        if obj_list is not None:
            return {i.id: i.name for i in obj_list}.get(id)


    def name_to_id(obj_list: list) -> dict:
        if obj_list is not None:
            return {i.name: i.id for i in obj_list}
        else:
            return {}
    return id_to_name, name_to_id


@app.cell
def _(client):
    # Set the list of projects available to the user
    projects = client.list_projects()
    projects.sort(key=lambda i: i.name)
    return (projects,)


@app.cell
def _(id_to_name, mo, name_to_id, projects, query_params):
    # Let the user select which project to get data from
    project_ui = mo.ui.dropdown(
        label="Select Project:",
        value=id_to_name(projects, query_params.get("project")),
        options=name_to_id(projects),
        on_change=lambda i: query_params.set("project", i)
    )
    project_ui
    return (project_ui,)


@app.cell
def _(cirro_dataset_type_filter, client, mo, project_ui):
    # Stop if the user has not selected a project
    mo.stop(project_ui.value is None)

    # Get the list of datasets available to the user
    # Filter the list of datasets by type (process_id)
    datasets = [
        dataset
        for dataset in client.get_project_by_id(project_ui.value).list_datasets()
        if dataset.process_id in cirro_dataset_type_filter
    ]
    return (datasets,)


@app.cell
def _(datasets, id_to_name, mo, name_to_id, query_params):
    # Let the user select which dataset to get data from
    dataset_ui = mo.ui.dropdown(
        label="Select Dataset:",
        value=id_to_name(datasets, query_params.get("dataset")),
        options=name_to_id(datasets),
        on_change=lambda i: query_params.set("dataset", i)
    )
    dataset_ui
    return (dataset_ui,)


@app.cell
def _(DataPortalDataset, dataset_ui, get_client, lru_cache, project_ui):
    @lru_cache
    def get_dataset(project_id: str, dataset_id: str) -> DataPortalDataset:
        _client = get_client()
        if _client is None:
            return
        return (
            _client
            .get_project_by_id(project_ui.value)
            .get_dataset_by_id(dataset_ui.value)
        )
    return (get_dataset,)


@app.cell
def _(dataset_ui, get_dataset, mo, project_ui):
    # Stop if the user has not selected a dataset
    mo.stop(dataset_ui.value is None)

    # Get the dataset object
    dataset = get_dataset(project_ui.value, dataset_ui.value)
    return (dataset,)


@app.cell
def _(np, pd):
    def calc_clr(vals: pd.Series):
        """Calculate the Centered Log Ratio for a vector of counts."""
        nonzero_vals = vals.loc[vals > 0]
        log_values = nonzero_vals.apply(np.log10)
        gmean = log_values.mean()
        return log_values / gmean
    return (calc_clr,)


@app.cell
def _(DataPortalDataset, Dict, List, defaultdict, mo, np, pd):
    # Read all of the data as a single object
    class Data:

        _ds: DataPortalDataset
        data: Dict[str, pd.DataFrame]
        # Key differential peaks by mark, caller, and comparison
        _differential_peaks: Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
        # Key peak annotations by mark and caller
        _peak_annotations = Dict[str, Dict[str, pd.DataFrame]]

        # All chromosomes
        _all_chromosomes: set

        def __init__(self, ds: DataPortalDataset):

            self._ds = ds

            self.read_data()
            self.parse_differential_peaks()
            self._all_chromosomes = set([])
            self.parse_peak_annotations()

        def read_data(self):
            """Read all of the CSVs in the dataset to the self.data object (keyed on filepath)."""
            self.data = dict()

            # Get the list of files that will be read in
            files_to_read = [
                file
                for file in self._ds.list_files()
                if (
                    # file.name.endswith(".csv") or
                    # file.name.endswith(".csv.gz") or
                    # file.name.endswith(".annotation.txt") or
                    # file.name.endswith("rawCounts.txt")
                    (file.name.endswith(".txt") and file.name.startswith("data/Analysis/08") and "_vs_" in file.name)
                    or
                    (file.name.endswith(".txt") and "_annotation" in file.name)
                )
            ]

            # Make a progress bar
            for file in mo.status.progress_bar(
                files_to_read,
                title="Loading Data",
                remove_on_exit=True,
                subtitle="Reading all CSVs/TSVs in dataset"
            ):
                key = (
                    file.name[len("data/"):]
                    if file.name.startswith("data/")
                    else file.name
                )
                try:
                    self.data[key] = file.read_csv(
                        sep=(
                            "," if "csv"
                            in file.name
                            else (
                                " " if file.name.endswith("rawCounts.txt")
                                else "\t"
                            )
                        )
                    )
                except Exception as e:
                    print(f"Problem reading {file.name}")
                    raise e

        def parse_peak_annotations(self):
            """Key the peak annotations by mark and caller."""
            self._peak_annotations = defaultdict(lambda: dict())

            prefix_lower = "analysis/07_consensus_peaks_by_target/_annotation/"
            suffix_lower = ".annotation.txt"
            self._all_chromosomes = set([])
            for key, df in self.data.items():
                if key.lower().startswith(prefix_lower) and key.lower().endswith(suffix_lower):
                    mark, caller = key[len(prefix_lower):-len(suffix_lower)].split(".", 1)
                    self._peak_annotations[mark][caller] = (
                        df
                        .set_index(df.columns.values[0])
                        .assign(
                            Middle=lambda d: d.apply(lambda r: np.mean([r['Start'], r['End']]), axis=1)
                        )
                    )
                    self._all_chromosomes |= set(df["Chr"].tolist())

        def peak_annotation(self, mark: str, caller: str):
            return self._peak_annotations.get(mark, {}).get(caller)

        def parse_differential_peaks(self):
            """Key the differential peak results, key by mark, caller, and comparison."""
            self._differential_peaks = defaultdict(lambda: defaultdict(lambda: dict()))

            prefix_lower = "analysis/08_differential_peaks/"
            for key, df in self.data.items():
                if key.lower().startswith(prefix_lower):
                    mark, caller = key[len(prefix_lower):].split("/")[0].split(".", 1)
                    comparison = key[len(f"{prefix_lower}{mark}.{caller}/"):].split("/")[0]
                    self._differential_peaks[mark][caller][comparison] = (
                        df
                        .assign(
                            neg_log10_pvalue=df["PValue"].apply(np.log10) * -1,
                            signed_log10_pvalue=lambda d: d.apply(lambda r: r["neg_log10_pvalue"] * (1 if r["logFC"] < 0 else -1), axis=1)
                        )
                        .sort_values(by="PValue")
                    )

        def marks(self) -> List[str]:
            return list(self._differential_peaks.keys())

        def callers(self) -> List[str]:
            return list(set([
                i
                for mark in self.marks()
                for i in list(self._differential_peaks[mark].keys())
            ]))

        def comparisons(self, mark: str, caller: str) -> List[str]:
            return list(self._differential_peaks[mark][caller].keys())

        def differential_peak(self, mark: str, caller: str, comparison: str) -> pd.DataFrame:
            return self._differential_peaks[mark][caller][comparison]

        def all_chromosomes(self):
            """Helper method to get a list of chromosomes in the genome."""
            return sorted(list(self._all_chromosomes))
    return (Data,)


@app.cell
def _(Data, dataset):
    data = Data(dataset)
    return (data,)


@app.cell
def _(data, mo):
    # Let the user pick a chromosome to display
    # Pick the start and end positions
    select_pos_ui = mo.md("""
    ### View Genomic Region

    {mark}

    {caller}

    {chr}

    {start}

    {stop}
    """).batch(
        mark = mo.ui.dropdown(
            label="Select Mark:",
            options=data.marks(),
            value=data.marks()[0]
        ),
        caller=mo.ui.dropdown(
            label="Select Caller:",
            options=data.callers(),
            value=data.callers()[0]
        ),
        chr=mo.ui.dropdown(
            label="Select Chromosome",
            options=data.all_chromosomes(),
            value="chr1"
        ),
        start=mo.ui.number(
            label="Start Position:",
            start=1,
            value=3000000
        ),
        stop=mo.ui.number(
            label="Start Position:",
            start=1,
            value=6000000
        )
    )
    select_pos_ui
    return (select_pos_ui,)


@app.cell
def _(data, select_pos_ui):
    def display_genomic_region(
        mark: str,
        caller: str,
        chr: str,
        start: int,
        stop: int
    ):
        # Get all of the peaks in the selected region
        all_peaks = data.peak_annotation(mark, caller)

        # Filter to the peaks within the window
        window_info = all_peaks.loc[
            all_peaks.apply(
                lambda r: (
                    r["Chr"] == chr
                    and
                    r["Start"] >= start
                    and
                    r["End"] <= stop
                ),
                axis=1
            )
        ].sort_values(by="Middle")

        return window_info

        # # Merge with the differential peaks information
        # merged = window_info.merge(data.differential_peak(mark, caller, comparison), left_index=True, right_on="conp.id")

        # # Add the label that will go into the plot
        # annotation_kws = [
        #     'Chr',
        #     'Start',
        #     'End',
        #     'Strand',
        #     'Annotation',
        #     'Distance to TSS',
        #     'Gene Name',
        #     'Gene Type',
        #     'logFC',
        #     'PValue',
        #     'FDR',
        #     'is.sig',
        #     'is.sig2'
        # ]
        # merged = merged.assign(
        #     annotation=merged.apply(
        #         lambda r: '<br>'.join([f"{kw}: {r[kw]}" for kw in annotation_kws]),
        #         axis=1
        #     )
        # )


    display_genomic_region(**select_pos_ui.value)
    return (display_genomic_region,)


@app.cell
def _():
    return


@app.cell
def _():
    # # Use state elements to keep the same selections even if one changes
    # get_mark, set_mark = mo.state(None)
    # get_caller, set_caller = mo.state(None)
    # get_comparison, set_comparison = mo.state(None)
    return


@app.cell
def _():
    # def dropdown_options(label: str, options: List[str], value: str, on_change: callable):
    #     return dict(
    #         label=label,
    #         options=options,
    #         value=value if value in options else None,
    #         on_change=on_change
    #     )
    return


@app.cell
def _():
    return


@app.cell
def _():
    # mo.stop(select_mark_ui.value is None)
    # selected_mark = select_mark_ui.value
    # callers = data.callers(selected_mark)

    # # Ask the user which caller to use
    # select_caller_ui = mo.ui.dropdown(**dropdown_options(
    #     label="Select Caller:",
    #     options=callers,
    #     value=get_caller(),
    #     on_change=set_caller
    # ))
    # select_caller_ui
    return


@app.cell
def _():
    # mo.stop(select_caller_ui.value is None)
    # selected_caller = select_caller_ui.value
    # comparisons = data.comparisons(selected_mark, selected_caller)

    # # Ask the user which comparison to view
    # select_comparison_ui = mo.ui.dropdown(**dropdown_options(
    #     label="Select Comparison:",
    #     options=comparisons,
    #     value=get_comparison(),
    #     on_change=set_comparison
    # ))
    # select_comparison_ui
    return


@app.cell
def _():
    # mo.stop(select_comparison_ui.value is None)
    # selected_comparison = select_comparison_ui.value

    # # Get the DataFrame with the comparison
    # comp_df = data.differential_peak(
    #     selected_mark,
    #     selected_caller,
    #     selected_comparison
    # )
    return


@app.cell
def _():
    # # Make a volcano plot
    # def make_volcano(comp_df: pd.DataFrame):

    #     fig = px.scatter(
    #         comp_df,
    #         x="logFC",
    #         y="neg_log10_pvalue",
    #         template="simple_white",
    #         hover_name="conp.id",
    #         hover_data=["sample.groups", "FDR"],
    #         size="logCPM",
    #         color="is.sig",
    #         labels=dict(
    #             logFC="Fold Change (log)",
    #             neg_log10_pvalue="pvalue (-log10)",
    #             logCPM="CPM (log)",
    #             **{
    #                 "is.sig": "Is Sig."
    #             }
    #         )
    #     )

    #     return fig

    # make_volcano(comp_df)
    return


@app.cell
def _():
    # # If there is more than one comparison available
    # mo.stop(len(comparisons) <= 1)

    # select_contrast_ui = mo.md("""
    # {comp}
    # {by}
    # """).batch(
    #     comp=mo.ui.dropdown(
    #         label="Contrast With:",
    #         options=[v for v in comparisons if v != selected_comparison],
    #         value=[v for v in comparisons if v != selected_comparison][0]
    #     ),
    #     by=mo.ui.dropdown(
    #         label="Contrast Using:",
    #         options=["Signed log10(pvalue)", "Fold Change (log2)"],
    #         value="Signed log10(pvalue)"
    #     )
    # )
    # select_contrast_ui
    return


@app.cell
def _():
    # mo.stop(select_contrast_ui.value['comp'] is None)
    # selected_contrast = select_contrast_ui.value["comp"]

    # # Get the DataFrame with the contrast
    # contrast_df = data.differential_peak(
    #     selected_mark,
    #     selected_caller,
    #     selected_contrast
    # )
    return


@app.cell
def _():
    # def make_contrast_plot(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str, by: str):
    #     """
    #     Constrast two different enrichment analyses.
    #     """
    #     by_labels = {"Signed log10(pvalue)": "signed_log10_pvalue", "Fold Change (log2)": "logFC"}
    #     index_cols = ["start", "end", "sample.groups", "npeak", "length", "chrom"]
    #     merged = df1.merge(
    #         df2.drop(
    #             columns=index_cols
    #         ),
    #         on="conp.id",
    #         suffixes=(f" {label1}", f" {label2}"),
    #         how="inner"
    #     )

    #     # Calculate the average CPM
    #     merged = merged.assign(**{
    #         "logCPM (mean)": (
    #             merged
    #             .reindex(columns=[f"logCPM {label1}", f"logCPM {label2}"])
    #             .mean(axis=1)
    #         )
    #     })

    #     xcol = f"{by_labels[by]} {label1}"
    #     ycol = f"{by_labels[by]} {label2}"

    #     # Make the plot
    #     fig = px.scatter(
    #         merged,
    #         x=xcol,
    #         y=ycol,
    #         hover_name="conp.id",
    #         hover_data=index_cols + [
    #             f"{prefix} {suffix}"
    #             for suffix in [label1, label2]
    #             for prefix in ["is.sig"]
    #         ],
    #         size="logCPM (mean)",
    #         labels={
    #             xcol: f"{by} - {label1}",
    #             ycol: f"{by} - {label2}",
    #         },
    #         template="simple_white"
    #     )
    #     return fig


    # make_contrast_plot(
    #     comp_df,
    #     contrast_df,
    #     selected_comparison,
    #     selected_contrast,
    #     select_contrast_ui.value["by"]
    # )
    return


@app.cell
def _():
    # # Let the user select a peak which can be inspected for its neighborhood in the genome
    # select_peak_ui = mo.md("""
    # {peak}

    # {window_size}
    # """).batch(
    #     peak=mo.ui.dropdown(
    #         label="Inspect Peak:",
    #         options=comp_df["conp.id"].sort_values().tolist(),
    #         value=comp_df["conp.id"].values[0]
    #     ),
    #     window_size=mo.ui.number(
    #         label="Window Size (bp):",
    #         start=10000,
    #         value=10000000,
    #         step=1000
    #     )
    # )
    # select_peak_ui
    return


@app.cell
def _():
    # def display_window(mark: str, caller: str, comparison: str, peak: str, window_size: int):
    #     # Get all of the peaks for this mark and caller
    #     all_peaks = data.peak_annotation(mark, caller)

    #     # Get the info for the selected peak
    #     peak_info = all_peaks.loc[peak]

    #     # Get the middle of the peak
    #     peak_pos = np.mean([peak_info['Start'], peak_info['End']])

    #     # Calculate the window limits
    #     window_start = int(peak_pos - (window_size / 2))
    #     window_end = int(peak_pos + (window_size / 2))

    #     # Filter to the peaks within the window
    #     window_info = all_peaks.loc[
    #         all_peaks.apply(
    #             lambda r: (
    #                 r["Chr"] == peak_info["Chr"]
    #                 and
    #                 r["Start"] > window_start
    #                 and
    #                 r["End"] < window_end
    #             ),
    #             axis=1
    #         )
    #     ].sort_values(by="Middle")

    #     # Merge with the differential peaks information
    #     merged = window_info.merge(data.differential_peak(mark, caller, comparison), left_index=True, right_on="conp.id")

    #     # Add the label that will go into the plot
    #     annotation_kws = [
    #         'Chr',
    #         'Start',
    #         'End',
    #         'Strand',
    #         'Annotation',
    #         'Distance to TSS',
    #         'Gene Name',
    #         'Gene Type',
    #         'logFC',
    #         'PValue',
    #         'FDR',
    #         'is.sig',
    #         'is.sig2'
    #     ]
    #     merged = merged.assign(
    #         annotation=merged.apply(
    #             lambda r: '<br>'.join([f"{kw}: {r[kw]}" for kw in annotation_kws]),
    #             axis=1
    #         )
    #     )

    #     fig = go.Figure(layout=dict(template="simple_white"))

    #     # Add a trace for the annotation
    #     fig.add_scatter(
    #         x=merged["Middle"],
    #         y=[0 for _ in merged.index.values],
    #         name="Annotation:",
    #         mode="none",
    #         text=merged["annotation"],
    #         hovertemplate="%{text}<extra></extra>"
    #     )

    #     # Plot the sequencing depth per sample
    #     sample_prefix = "FPKM.TMMnormalized."
    #     samples = [cname for cname in merged.columns if cname.startswith(sample_prefix)]
    #     for sample, fpkm in merged.reindex(columns=samples).items():
    #         fig.add_scatter(
    #             x=merged["Middle"],
    #             y=fpkm,
    #             name=sample[len(sample_prefix):]
    #         )
    #     fig.update_layout(
    #         xaxis=dict(
    #             title=dict(
    #                 text=peak_info["Chr"]
    #             )
    #         ),
    #         yaxis=dict(
    #             title=dict(
    #                 text="Sequencing Depth (FPKM)"
    #             )
    #         ),
    #         hovermode="x unified"
    #     )

    #     return merged

    #     return fig
    return


@app.cell
def _():
    # display_window(mark=selected_mark, caller=selected_caller, comparison=selected_comparison, **select_peak_ui.value)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
