import marimo

__generated_with = "0.11.0"
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
        from typing import Dict, Optional
        import plotly.express as px
        import pandas as pd
        import numpy as np
        from functools import lru_cache
        import base64
        from urllib.parse import quote_plus

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
        Optional,
        Queue,
        StringIO,
        base64,
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
    return (
        domain_to_name,
        name_to_domain,
        tenants_by_domain,
        tenants_by_name,
    )


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
        value=id_to_name(datasets, query_params.get("dataset")),
        options=name_to_id(datasets),
        on_change=lambda i: query_params.set("dataset", i)
    )
    dataset_ui
    return (dataset_ui,)


@app.cell
def _(client, dataset_ui, mo, project_ui):
    # Stop if the user has not selected a dataset
    mo.stop(dataset_ui.value is None)

    # Get the dataset object
    dataset = (
        client
        .get_project_by_id(project_ui.value)
        .get_dataset_by_id(dataset_ui.value)
    )
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
def _(DataPortalDataset, Dict, Tuple, calc_clr, mo, pd):
    # Read all of the data as a single object
    class Data:

        _ds: DataPortalDataset
        data: Dict[str, pd.DataFrame]
        umaps: Dict[Tuple(str, str), pd.DataFrame]
        counts: Dict[Tuple(str, str), pd.DataFrame]
        clr: Dict[Tuple(str, str), pd.DataFrame]
        all_marks = []
        all_callers = []

        def __init__(self, ds: DataPortalDataset):

            self._ds = ds

            self.read_data()
            self.merge_umap_tables()
            self.format_counts()

        def format_counts(self):
            self.counts = {}
            self.clr = {}
            key_prefix = "analysis/08_differential_peaks/"
            key_suffix = "/rawcounts.txt"
            keys_to_parse = [
                key
                for key in self.data.keys()
                if key.startswith(key_prefix) and key.endswith(key_suffix)
            ]
            for key in mo.status.progress_bar(keys_to_parse, title="Formatting Counts", remove_on_exit=True):
                mark, caller = key[len(key_prefix):-len(key_suffix)].split('.')
                df = self.data[key]

                index_cols = ["chrom", "start", "end", "conp.id", "sample.groups", "npeak", "length"]
                for cname in index_cols:
                    assert cname in df.columns.values, f"Expected to find {cname} in {key}"

                    # Save the counts
                    self.counts[(mark, caller)] = df.set_index(index_cols)

                    # Also compute the centered log ratio
                    self.clr[(mark, caller)] = self.counts[(mark, caller)].apply(calc_clr).fillna(0)

        def merge_umap_tables(self):
            # Keep track of all of the UMAP merged tables
            self.umaps = dict()
            self.all_callers = []
            self.all_marks = []

            keys_to_merge = [
                key
                for key in self.data.keys()
                if key.startswith("umaps/")
            ]
            for key in mo.status.progress_bar(keys_to_merge, title="Formatting UMAPs", remove_on_exit=True):
                mark, caller = key[len("umaps/"):-len(".csv")].split('.')
                if mark not in self.all_marks:
                    self.all_marks.append(mark)
                if caller not in self.all_callers:
                    self.all_callers.append(caller)

                analysis_key = f"analysis/07_consensus_peaks_by_target/_annotation/{mark}.{caller}.annotation.txt"
                assert analysis_key in self.data, f"Could not find analysis ({analysis_key})"
                merged_table = self.data[key].merge(
                    self.data[analysis_key],
                    left_on=self.data[key].columns.values[0],
                    right_on=self.data[analysis_key].columns.values[0]
                )
                self.umaps[(mark, caller)] = merged_table                

        def read_data(self):
            """Read all of the CSVs in the dataset to the self.data object (keyed on filepath)."""
            self.data = dict()

            # Get the list of files that will be read in
            files_to_read = [
                file
                for file in self._ds.list_files()
                if (
                    file.name.endswith(".csv") or
                    file.name.endswith(".csv.gz") or
                    file.name.endswith(".annotation.txt") or
                    file.name.endswith("rawCounts.txt")
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
                    (file.name[len("data/"):] if file.name.startswith("data/") else file.name)
                    .lower()
                )
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
    return (Data,)


@app.cell
def _(Data, dataset):
    data = Data(dataset)
    return (data,)


@app.cell
def _(data, mo):
    compare_samples_ui = (
        mo.md("""
    #### Compare Peaks Across Samples
    {mark}

    {caller}
        """)
        .batch(
            mark=mo.ui.dropdown(
                data.all_marks,
                label="Select Mark:"
            ),
            caller=mo.ui.dropdown(
                data.all_callers,
                label="Select Caller:"
            )
        )
    )
    compare_samples_ui
    return (compare_samples_ui,)


@app.cell
def _(compare_samples_ui, data, mo):
    # When the user selects the mark and the caller, get the options for what
    # samples have counts available
    mo.stop(compare_samples_ui.value["mark"] is None or compare_samples_ui.value["caller"] is None)
    compare_samples_options = data.clr[(
        compare_samples_ui.value["mark"],
        compare_samples_ui.value["caller"]
    )].columns.values
    return (compare_samples_options,)


@app.cell
def _(compare_samples_options, mo):
    # Ask the user what samples to compare
    compare_samples_groups_ui = (
        mo.md("""
    {groupA}

    {groupB}
    """)
        .batch(
            groupA=mo.ui.multiselect(
                compare_samples_options,
                label="Select Group A:"
            ),
            groupB=mo.ui.multiselect(
                compare_samples_options,
                label="Select Group B:"
            )
        )
    )
    compare_samples_groups_ui
    return (compare_samples_groups_ui,)


@app.cell
def _(compare_samples_groups_ui, mo):
    group_list_a = '\n - '.join(compare_samples_groups_ui.value["groupA"])
    group_list_b = '\n - '.join(compare_samples_groups_ui.value["groupB"])
    mo.md(f"""
    Group A:

    - {group_list_a}

    Group B:

    - {group_list_b}

    """)
    return


@app.cell
def _(mo):
    with mo.status.spinner("Loading Dependencies:"):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from scipy import stats
    return plt, sns, stats


@app.cell
def _(compare_samples_groups_ui, data, np, pd, stats):
    def calc_mwu_p(vals: pd.Series):
        # Get the samples in each group
        groupA = compare_samples_groups_ui.value["groupA"]
        groupB = compare_samples_groups_ui.value["groupB"]
        valsA = vals.loc[groupA]
        valsB = vals.loc[groupB]
        mwu = stats.mannwhitneyu(valsA, valsB)
        return mwu.pvalue

    def calc_ttest_p(vals: pd.Series):
        # Get the samples in each group
        groupA = compare_samples_groups_ui.value["groupA"]
        groupB = compare_samples_groups_ui.value["groupB"]
        valsA = vals.loc[groupA]
        valsB = vals.loc[groupB]
        mwu = stats.ttest_ind(valsA, valsB)
        return mwu.pvalue

    def compare_samples_prep_data(mark: str, caller: str, groupA: list, groupB: list):

        if mark is None or caller is None:
            return
        if len(groupA) == 0 or len(groupB) == 0:
            return

        # Get the table of counts (CLR transform)
        df = data.clr[(mark, caller)]

        # Split up the counts for each group
        dfA = df.reindex(columns=groupA)
        dfB = df.reindex(columns=groupB)

        # Get the mean value for each and calculate the log-fold-change
        merged = (
            pd.DataFrame(dict(
                meanA=dfA.mean(axis=1),
                meanB=dfB.mean(axis=1),
                # # Use Mann-Whitney U to compare both populations
                # mwu_p=pd.concat([dfA, dfB], axis=1).apply(calc_mwu_p, axis=1),
                # Also use a t-test
                ttest_p=pd.concat([dfA, dfB], axis=1).apply(calc_ttest_p, axis=1),
            ))
            .query(
                "meanA > 0 or meanB > 0"
            )
            .assign(
                log10_fold_change=lambda d: d['meanA'] - d['meanB'],
                mean_abund=lambda d: d[['meanA', 'meanB']].mean(axis=1),
            )
        )
        merged = merged.assign(
            # mwu_neg_log10_p=(-1 * (merged["mwu_p"].apply(np.log10))).apply(np.abs),
            ttest_neg_log10_p=(-1 * (merged["ttest_p"].apply(np.log10))).apply(np.abs),
        )

        return merged.reset_index()

    return calc_mwu_p, calc_ttest_p, compare_samples_prep_data


@app.cell
def _(
    compare_samples_groups_ui,
    compare_samples_prep_data,
    compare_samples_ui,
    mo,
):
    with mo.status.spinner("Preparing Data:"):
        plot_df = compare_samples_prep_data(**compare_samples_ui.value, **compare_samples_groups_ui.value)
    mo.stop(plot_df is None)
    return (plot_df,)


@app.cell
def _(mo, pd, plot_df, px):
    def compare_samples_plot(df: pd.DataFrame):
        # Make a plot of the mean abundance and log fold change
        fig = px.scatter(
            df.sort_values(by="ttest_neg_log10_p", ascending=False).head(1000),
            size="mean_abund",
            color="mean_abund",
            x="log10_fold_change",
            y="ttest_neg_log10_p",
            template="simple_white",
            hover_name="conp.id",
            hover_data=["chrom", "start", "end", "sample.groups", "ttest_p"],
            color_continuous_scale="bluered",
            labels=dict(
                mean_abund="Mean Abundance (CLR)",
                log10_fold_change="Fold Change (log10)",
                ttest_neg_log10_p="t-test p-vlue (-log10)",
                ttest_p="t-test p-value"
            )
        )

        return mo.ui.plotly(fig)


    compare_samples_plot_ui = compare_samples_plot(plot_df)
    compare_samples_plot_ui
    return compare_samples_plot, compare_samples_plot_ui


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
