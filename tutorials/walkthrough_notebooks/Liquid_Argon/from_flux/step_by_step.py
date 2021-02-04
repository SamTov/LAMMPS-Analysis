import shutil

from mdsuite.experiment.experiment import Experiment

if __name__ == '__main__':

    new_case = True

    if new_case:
        try:
            shutil.rmtree('Argon_94')
        except FileNotFoundError:
            pass
    argon = Experiment(analysis_name="Argon_94",
                             storage_path=".",
                             temperature=94.4,
                             time_step=2,
                             units='real')

    argon.add_data(trajectory_file='../flux_1.lmp_flux', file_format='lammps_flux', rename_cols={"Flux_Thermal":['flux[1]','flux[2]', "flux[3]"]})
    argon.run_computation('GreenKuboThermalConductivityFlux', data_range=10990, plot=True)
