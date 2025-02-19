import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium", app_title="Differential Expression Viewer")


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
                "boto3==1.36.3",
                "botocore==1.36.3"
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
def _(DataPortalDataset, Dict, mo, pd):
    # Read all of the data as a single object
    class Data:

        _ds: DataPortalDataset
        data: Dict[str, pd.DataFrame]

        def __init__(self, ds: DataPortalDataset):

            self._ds = ds

            self.read_data()

            self.key_umaps()

        def read_data(self):
            """Read all of the CSVs in the dataset to the self.data object (keyed on filepath)."""
            self.data = dict()

            # Get the list of files that will be read in
            files_to_read = [
                file
                for file in self._ds.list_files()
                if file.name.endswith(".csv") or file.name.endswith(".csv.gz")
            ]

            # Make a progress bar
            for file in mo.status.progress_bar(
                files_to_read,
                title="Loading Data",
                remove_on_exit=True,
                subtitle="Reading all CSVs in dataset"
            ):
                key = (
                    (file.name[len("data/"):] if file.name.startswith("data/") else file.name)
                    .lower()
                )
                self.data[key] = file.read_csv()
    return (Data,)


@app.cell
def _(Data, dataset):
    data = Data(dataset)
    return (data,)


@app.cell
def _(data, px):
    # Plot the consensus peaks

    fig = px.bar(
        data.data["csv/consensus_peaks.csv"],
        y="nconp",
        x="target",
        color="caller",
        title="Consensus Peaks",
        barmode="group",
        template="simple_white",
        labels=dict(nconp="Number of Peaks", caller="Caller", target="Target")
    )
    fig
    return (fig,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
