"""Get material properties."""

import os
import subprocess  # noqa: S404  # only used for hardcoded calls
import tempfile
from time import sleep


def get_thermal_conductivity(
    material: str, temperatures, workdir: os.PathLike, ees: os.PathLike, wait: float = 7
):
    """Get thermal conductivity."""

    files = {
        key: tempfile.NamedTemporaryFile(delete=False)
        for key in ["in", "out", "script"]
    }

    # write post material, number of runs, and average post temperatures to in.dat
    with files["in"] as f:
        f.write(
            f"{material} {len(temperatures)} {' '.join([str(t) for t in temperatures])}".encode()
        )

    get_thermal_conductivity_script = (
        f"$Import '{files['in'].name}' Material$ N T[1..N]\n\n"
        "Duplicate j=1,N\n"
        "    k[j] = Conductivity(Material$, T=T[j])\n"
        "End\n\n"
        # f"$Export '{files['out'].name}' k[1..N]"
        f"$Export '{files['out'].name}' k[1..N]"
    )

    with files["script"] as f:
        f.write(get_thermal_conductivity_script.encode())

    # Invoke EES to write thermal conductivities to out.dat given contents of in.dat
    subprocess.run(  # noqa: S603  # hardcoded
        [
            f"{ees}",
            f"{files['script'].name}",
            "/solve",
        ]
    )
    sleep(wait)  # Wait long enough for EES to finish

    # # EES should have written to out.dat
    # with files["out"] as f:
    #     k_str = f.read().decode().split("\t")
    #     thermal_conductivity = np.array(k_str, dtype=np.float64)

    for file in files.values():
        os.unlink(file.name)

    # return thermal_conductivity
