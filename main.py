import pprint
import array_illumination

if __name__ == '__main__':
    """What data are we processing?"""
    xPix, yPix, zPix, steps = 128, 128, 224, 224
    background_zPix = zPix
    extent = 5
    animate = False  # Temporarily set to "True" if you have to adjust 'extent'
    scan_type = 'dmd'
    scan_dimensions = (16, 14)
    preframes = 0
    num_harmonics = 3  # Default to 3, might have to lower to 2

    data_dir = 'D:\\Files\\OneDrive - stu.hit.edu.cn\\codes\\python\\MSIM-MPSS-tiff\\data\\241217-change_psf\\MSIM_lake_nb_blur=2'
    data_filenames_list = [
        'D:\\Files\\OneDrive - stu.hit.edu.cn\\codes\\python\\MSIM-MPSS-tiff\\data\\241217-change_psf\\MSIM_lake_nb_blur=2\\tubules488.tiff',
        'D:\\Files\\OneDrive - stu.hit.edu.cn\\codes\\python\\MSIM-MPSS-tiff\\data\\241217-change_psf\\MSIM_lake_nb_blur=2\\lake.tiff']
    lake_filename = 'D:\\Files\\OneDrive - stu.hit.edu.cn\\codes\\python\\MSIM-MPSS-tiff\\data\\241217-change_psf\\MSIM_lake_nb_blur=2\\lake.tiff'
    background_filename = 'D:\\Files\\OneDrive - stu.hit.edu.cn\\codes\\python\\MSIM-MPSS-tiff\\data\\241217-change_psf\\MSIM_lake_nb_blur=2\\background.tiff'

    use_all_lake_parameters = True

    print("Calibration source:", lake_filename)
    print("Background source:", background_filename)

    # 找到组晶格向量、偏移向量和位移向量
    print("\nDetecting illumination lattice parameters...")
    lattice_vectors, shift_vector, offset_vector, intensities_vs_galvo_position, background_frame \
        = array_illumination.get_lattice_vectors(
        filename_list=data_filenames_list,
        lake=lake_filename, bg=background_filename,
        use_lake_lattice=True,
        use_all_lake_parameters=use_all_lake_parameters,
        xPix=xPix, yPix=yPix, zPix=zPix, bg_zPix=background_zPix,
        preframes=preframes,
        extent=extent,  # Important to get this right.
        num_spikes=300,
        tolerance=3.5,
        num_harmonics=num_harmonics,
        outlier_phase=1.,
        calibration_window_size=10,
        scan_type=scan_type,
        scan_dimensions=scan_dimensions,
        verbose=True,  # Useful for debugging
        display=True,  # Useful for debugging
        animate=animate,  # Useful to see if 'extent' is right
        show_interpolation=False,  # Fairly low-level debugging
        show_calibration_steps=False,  # Useful for debugging
        show_lattice=True)  # Very useful for checking validity
    print("Lattice vectors:")
    for v in lattice_vectors:
        print(v)
    print("Shift vector:")
    pprint.pprint(shift_vector)
    print("Initial position:")
    print(offset_vector)

    """Define a new Cartesian grid for Enderlein's trick:"""
    new_grid_xrange = 0, xPix - 1, 2 * xPix
    new_grid_yrange = 0, yPix - 1, 2 * yPix

    num_processes = 6
    for f in data_filenames_list:
        print(f)
    

        def profile_me():
            array_illumination.enderlein_image_parallel(
                data_filename=f,
                lake_filename=lake_filename,
                background_filename=background_filename,
                xPix=xPix, yPix=yPix, zPix=zPix, steps=steps, preframes=preframes,
                lattice_vectors=lattice_vectors,
                offset_vector=offset_vector,
                shift_vector=shift_vector,
                new_grid_xrange=new_grid_xrange, new_grid_yrange=new_grid_yrange,
                num_processes=num_processes,
                window_footprint=10,
                aperture_size=3,
                make_widefield_image=False,
                make_confocal_image=False,  # Broken, for now
                verbose=True,
                show_steps=False,  # For debugging
                show_slices=False,  # For debugging
                intermediate_data=False,  # Memory hog, leave 'False'
                normalize=False,  # Of uncertain merit, leave 'False' probably
                display=False
            )


        if num_processes == 1:
            import cProfile
            import pstats

            cProfile.run('profile_me()', 'profile_results')

            try:
                p = pstats.Stats('profile_results')
                p.strip_dirs().sort_stats(-1).print_stats()
                p.sort_stats('cumulative').print_stats(20)
            except ImportError:
                pass
        else:
            profile_me()

    array_illumination.join_enderlein_images(
        data_filenames_list,
        new_grid_xrange, new_grid_yrange,
        join_widefield_images=False)
