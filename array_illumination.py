import os, sys, pickle, pprint, subprocess, time, random
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, interpolation
from scipy.signal.windows import hann, gaussian
import cv2


def get_lattice_vectors(
        filename_list=None,
        lake=None,
        bg=None,
        use_lake_lattice=False,
        use_all_lake_parameters=False,
        xPix=512,
        yPix=512,
        zPix=201,
        bg_zPix=200,
        preframes=0,
        extent=15,
        num_spikes=300,
        tolerance=3.,
        num_harmonics=3,
        outlier_phase=1.,
        calibration_window_size=10,
        scan_type='visitech',
        scan_dimensions=None,
        verbose=False,
        display=False,
        animate=False,
        show_interpolation=False,
        show_calibration_steps=False,
        show_lattice=False,
        record_parameters=True):
    if len(filename_list) < 1:
        raise UserWarning('Filename list is empty.')

    """Given a swept-field confocal image stack, finds
    the basis vectors of the illumination lattice pattern."""
    if lake is not None:
        print(" Detecting lake illumination lattice parameters...")
        lake_lattice_vectors, lake_shift_vector, lake_offset_vector = get_lattice_vectors(
            filename_list=[lake], lake=None, bg=None,
            xPix=xPix, yPix=yPix, zPix=zPix, preframes=preframes,
            extent=extent,
            num_spikes=num_spikes,
            tolerance=tolerance,
            num_harmonics=num_harmonics,
            outlier_phase=outlier_phase,
            scan_type=scan_type,
            scan_dimensions=scan_dimensions,
            verbose=verbose,
            display=display,
            animate=animate,
            show_interpolation=show_interpolation,
            show_lattice=show_lattice)
        print("Lake lattice vectors:")
        for v in lake_lattice_vectors:
            print(v)
        print("Lake shift vector:")
        pprint.pprint(lake_shift_vector)
        print("Lake initial position:")
        print(lake_offset_vector)
        print(" Detecting sample illumination parameters...")
    if use_all_lake_parameters and (lake is not None):
        direct_lattice_vectors = lake_lattice_vectors
        shift_vector = lake_shift_vector
        corrected_shift_vector = lake_shift_vector
        offset_vector = lake_offset_vector
    elif use_lake_lattice and (lake is not None):
        """Keep the lake's lattice and crude shift vector"""
        direct_lattice_vectors = lake_lattice_vectors
        shift_vector = lake_shift_vector
        """The offset vector is now cheap to compute"""
        first_image_proj = np.zeros((xPix, yPix), dtype=np.float64)
        print("Computing projection of first image...")
        for i, f in enumerate(filename_list):
            im = load_image_slice(filename=f, xPix=xPix, yPix=yPix, preframes=preframes, which_slice=0)
            first_image_proj = np.where(
                im > first_image_proj, im, first_image_proj)
            sys.stdout.write('\rProjecting image %i' % (i))
            sys.stdout.flush()
        offset_vector = get_offset_vector(
            image=first_image_proj,
            direct_lattice_vectors=direct_lattice_vectors,
            verbose=verbose, display=display,
            show_interpolation=show_interpolation)
        print(offset_vector)
        """And the shift vector is cheap to correct"""
        last_image_proj = np.zeros((xPix, yPix), dtype=np.float64)
        print("Computing projection of first image...")
        for f in filename_list:
            im = load_image_slice(filename=f, xPix=xPix, yPix=yPix, preframes=preframes, which_slice=zPix - 1)
            last_image_proj = np.where(
                im > last_image_proj, im, last_image_proj)
            sys.stdout.write('\rProjecting image %i' % (i))
            sys.stdout.flush()
        corrected_shift_vector, final_offset_vector = get_precise_shift_vector(
            direct_lattice_vectors, shift_vector, offset_vector,
            last_image_proj, zPix, scan_type, verbose)
        print(corrected_shift_vector, final_offset_vector)
    else:
        if len(filename_list) > 1:
            raise UserWarning(
                "Processing multiple files without a lake calibration" +
                " is not supported.")
        """We do this the hard way"""
        image_data = load_image_data(filename_list[0], xPix=xPix, yPix=yPix, zPix=zPix, preframes=preframes)
        fft_data_folder, fft_abs, fft_avg = get_fft_abs(filename_list[0], image_data)  # DC term at center
        filtered_fft_abs = spike_filter(fft_abs)

        """Find candidate spikes in the Fourier domain"""
        coords = find_spikes(fft_abs, filtered_fft_abs, extent=extent, num_spikes=num_spikes, display=display,
                             animate=animate)
        """Use these candidate spikes to determine the
        Fourier-space lattice"""
        if verbose:
            print("Finding Fourier-space lattice vectors...")
        basis_vectors = get_basis_vectors(fft_abs, coords, extent=extent, tolerance=tolerance,
                                          num_harmonics=num_harmonics, verbose=verbose)
        if verbose:
            print("Fourier-space lattice vectors:")
            for v in basis_vectors:
                print(v, "(Magnitude", np.sqrt((v ** 2).sum()), ")")
        """Correct the Fourier-space vectors by constraining their
        sum to be zero"""
        error_vector = sum(basis_vectors)
        corrected_basis_vectors = [v - ((1. / 3.) * error_vector) for v in basis_vectors]
        if verbose:
            print("Fourier-space lattice vector triangle sum:", error_vector)
            print("Corrected Fourier-space lattice vectors:")
            for v in corrected_basis_vectors:
                print(v)
        """Determine the real-space lattice from the Fourier lattice"""
        area = np.cross(corrected_basis_vectors[0], corrected_basis_vectors[1])
        rotate_90 = ((0., -1.), (1., 0.))
        direct_lattice_vectors = [np.dot(v, rotate_90) * fft_abs.shape / area for v in corrected_basis_vectors]
        if verbose:
            print("Real-space lattice vectors:")
            for v in direct_lattice_vectors:
                print(v, "(Magnitude", np.sqrt((v ** 2).sum()), ")")
            print("Lattice vector triangle sum:")
            print(sum(direct_lattice_vectors))
            print("Unit cell area: (%0.2f)^2 square pixels" % (
                np.sqrt(np.abs(np.cross(direct_lattice_vectors[0], direct_lattice_vectors[1])))))

        """Use the Fourier lattice and the image data to measure shift and offset"""
        offset_vector = get_offset_vector(
            image=image_data[0, :, :],
            direct_lattice_vectors=direct_lattice_vectors,
            verbose=verbose, display=display,
            show_interpolation=show_interpolation)

        shift_vector = get_shift_vector(
            corrected_basis_vectors, fft_data_folder, filtered_fft_abs,
            num_harmonics=num_harmonics, outlier_phase=outlier_phase,
            verbose=verbose, display=display,
            scan_type=scan_type, scan_dimensions=scan_dimensions)

        corrected_shift_vector, final_offset_vector = get_precise_shift_vector(
            direct_lattice_vectors, shift_vector, offset_vector,
            image_data[-1, :, :], zPix, scan_type, verbose)

    if show_lattice:
        which_filename = 0
        while True:
            print("Displaying:", filename_list[which_filename])
            image_data = load_image_data(filename_list[which_filename], xPix=xPix, yPix=yPix, zPix=zPix,
                                         preframes=preframes)
            show_lattice_overlay(
                image_data, direct_lattice_vectors,
                offset_vector, corrected_shift_vector)
            if len(filename_list) > 1:
                which_filename = input(
                    "Display lattice overlay for which dataset? [done]:")
                try:
                    which_filename = int(which_filename)
                except ValueError:
                    if which_filename == '':
                        print("Done displaying lattice overlay.")
                        break
                    else:
                        continue
                if which_filename >= len(filename_list):
                    which_filename = len(filename_list) - 1
            else:
                break

    # image_data is large. Figures hold references to it, stinking up the place.
    if display or show_lattice:
        plt.close('all')
        import gc
        gc.collect()  # Actually required, for once!

    if record_parameters:
        params_file_path = os.path.join(os.path.dirname(filename_list[0]), 'parameters.txt')

        with open(params_file_path, 'w') as params:
            params.write("Direct lattice vectors: {}\n\n".format(repr(direct_lattice_vectors)))
            params.write("Corrected shift vector: {}\n\n".format(repr(corrected_shift_vector)))
            params.write("Offset vector: {}\n\n".format(repr(offset_vector)))
            try:
                params.write("Final offset vector: {}\n\n".format(repr(final_offset_vector)))
            except UnboundLocalError:
                params.write("Final offset vector: Not recorded\n\n")
            if lake is not None:
                params.write("Lake filename: {}\n\n".format(lake))

    if lake is None or bg is None:
        return direct_lattice_vectors, corrected_shift_vector, offset_vector
    else:
        intensities_vs_galvo_position, background_frame = spot_intensity_vs_galvo_position(lake, xPix, yPix, zPix,
                                                                                           preframes,
                                                                                           lake_lattice_vectors,
                                                                                           lake_shift_vector,
                                                                                           lake_offset_vector,
                                                                                           bg, bg_zPix,
                                                                                           window_size=calibration_window_size,
                                                                                           verbose=verbose,
                                                                                           show_steps=show_calibration_steps)
        return direct_lattice_vectors, corrected_shift_vector, offset_vector, intensities_vs_galvo_position, background_frame


def enderlein_image_parallel(
        data_filename, lake_filename, background_filename,
        xPix, yPix, zPix, steps, preframes,
        lattice_vectors, offset_vector, shift_vector,
        new_grid_xrange, new_grid_yrange,
        num_processes=1,
        window_footprint=10,
        aperture_size=3,
        make_widefield_image=True,
        make_confocal_image=False,  # Broken, for now
        verbose=True,
        show_steps=False,  # For debugging
        show_slices=False,  # For debugging
        intermediate_data=False,  # Memory hog, for stupid reasons, leave 'False'
        normalize=False,  # Of uncertain merit, leave 'False' probably
        display=False,
):
    input_arguments = locals()
    input_arguments.pop('num_processes')

    print("\nCalculating Enderlein image")
    basename = os.path.splitext(data_filename)[0]
    print(basename)

    enderlein_image_name = basename + '_enderlein_image.raw'

    if os.path.exists(enderlein_image_name):
        print("\nEnderlein image already calculated.")
        print("Loading", os.path.split(enderlein_image_name)[1])
        images = {}
        try:
            images['enderlein_image'] = np.fromfile(enderlein_image_name, dtype=float).reshape(new_grid_xrange[2],
                                                                                               new_grid_yrange[2])
        except ValueError:
            print("\n\nWARNING: the data file:")
            print(enderlein_image_name)
            print("may not be the size it was expected to be.\n\n")
            raise
    else:
        start_time = time.perf_counter()
        if num_processes == 1:
            images = enderlein_image_subprocess(**input_arguments)
        else:
            input_arguments['intermediate_data'] = False  # Difficult for parallel
            input_arguments['show_steps'] = False  # Difficult for parallel
            input_arguments['show_slices'] = False  # Difficult for parallel
            input_arguments['display'] = False  # Annoying for parallel
            input_arguments['verbose'] = False  # Annoying for parallel

            step_boundaries = list(range(0, steps, 10)) + [steps]
            step_boundaries = [(step_boundaries[i], step_boundaries[i + 1] - 1) for i in
                               range(len(step_boundaries) - 1)]
            running_processes = {}
            first_harvest = True
            random_prefix = '%06i_' % (random.randint(0, 999999))
            while len(running_processes) > 0 or len(step_boundaries) > 0:
                """Load up the subprocesses"""
                while (len(running_processes) < num_processes and
                       len(step_boundaries) > 0):
                    sb = step_boundaries.pop(0)
                    input_arguments['start_frame'], input_arguments['end_frame'] = sb
                    output_filename = (random_prefix + '%i_%i_intermediate_data.temp' % sb)
                    sys.stdout.write("\rProcessing frames: " + repr(sb[0]) + '-' + repr(sb[1]) + ' ' * 10)
                    sys.stdout.flush()
                    command_string = """
import array_illumination, pickle
from numpy import array
input_arguments=%s
sub_images = array_illumination.enderlein_image_subprocess(**input_arguments)
pickle.dump(sub_images, open('%s', 'wb'), protocol=2)
""" % (repr(input_arguments), output_filename)
                    running_processes[output_filename] = subprocess.Popen(
                        [sys.executable, '-c %s' % command_string],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
                """Poke each subprocess, harvest the finished ones"""
                pop_me = []
                for f, proc in running_processes.items():
                    if proc.poll() is not None:  # Time to harvest
                        pop_me.append(f)
                        report = proc.communicate()
                        if report != (b'', b''):
                            print(report)
                            raise UserWarning("Problem with a subprocess.")
                        sub_images = pickle.load(open(f, 'rb'))
                        os.remove(f)

                        if first_harvest:
                            images = sub_images
                            first_harvest = False
                        else:
                            for k in images.keys():
                                images[k] += sub_images[k]
                for p in pop_me:  # Forget about the harvested processes
                    running_processes.pop(p)
                """Chill for a second"""
                time.sleep(0.2)
        end_time = time.perf_counter()
        print("Elapsed time: %0.2f seconds" % (end_time - start_time))
        images['enderlein_image'].tofile(enderlein_image_name)
        if make_widefield_image:
            images['widefield_image'].tofile(basename + '_widefield.raw')
        if make_confocal_image:
            images['confocal_image'].tofile(basename + '_confocal.raw')
    display = True
    if display:
        plt.figure()
        plt.imshow(images['enderlein_image'], interpolation='nearest', cmap="gray")
        plt.colorbar()
        plt.show()
    return images


def enderlein_image_subprocess(
        data_filename, lake_filename, background_filename,
        xPix, yPix, zPix, steps, preframes,
        lattice_vectors, offset_vector, shift_vector,
        new_grid_xrange, new_grid_yrange,
        start_frame=None, end_frame=None,
        window_footprint=10,
        aperture_size=3,
        make_widefield_image=True,
        make_confocal_image=False,  # Broken, for now
        verbose=True,
        show_steps=False,  # For debugging
        show_slices=False,  # For debugging
        intermediate_data=False,  # Memory hog, for stupid reasons. Leave 'False'
        normalize=False,  # Of uncertain merit, leave 'False' probably
        display=False,
):
    basename = os.path.splitext(data_filename)[0]
    enderlein_image_name = basename + '_enderlein_image.raw'
    lake_basename = os.path.splitext(lake_filename)[0]
    lake_intensities_name = lake_basename + '_spot_intensities.pkl'
    background_basename = os.path.splitext(background_filename)[0]
    background_name = background_basename + '_background_image.raw'

    intensities_vs_galvo_position = pickle.load(open(lake_intensities_name, 'rb'))
    background_directory_name = os.path.dirname(background_name)
    try:
        background_frame = np.fromfile(background_name).reshape(xPix, yPix).astype(float)
    except ValueError:
        print("\n\nWARNING: the data file:")
        print(background_name)
        print("may not be the size it was expected to be.\n\n")
        raise
    try:
        hot_pixels = np.fromfile(os.path.join(background_directory_name, 'hot_pixels.txt'), sep=', ')
    except:
        hot_pixels = None

    else:
        hot_pixels = hot_pixels.reshape(2, len(hot_pixels) // 2)

    if show_steps or show_slices: fig = plt.figure()
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = steps - 1
    new_grid_x = np.linspace(*new_grid_xrange)
    new_grid_y = np.linspace(*new_grid_yrange)
    enderlein_image = np.zeros((new_grid_x.shape[0], new_grid_y.shape[0]), dtype=np.float64)
    enderlein_normalization = np.zeros_like(enderlein_image)
    this_frames_enderlein_image = np.zeros_like(enderlein_image)
    this_frames_normalization = np.zeros_like(enderlein_image)
    if intermediate_data:
        cumulative_sum = np.memmap(
            basename + '_cumsum.raw', dtype=float, mode='w+',
            shape=(steps,) + enderlein_image.shape)
        processed_frames = np.memmap(
            basename + '_frames.raw', dtype=float, mode='w+',
            shape=(steps,) + enderlein_image.shape)
    if make_widefield_image:
        widefield_image = np.zeros_like(enderlein_image)
        widefield_coordinates = np.meshgrid(new_grid_x, new_grid_y)
        widefield_coordinates = (
            widefield_coordinates[0].reshape(
                new_grid_x.shape[0] * new_grid_y.shape[0]),
            widefield_coordinates[1].reshape(
                new_grid_x.shape[0] * new_grid_y.shape[0]))
    if make_confocal_image:
        confocal_image = np.zeros_like(enderlein_image)
    enderlein_normalization.fill(1e-12)
    aperture = gaussian(2 * window_footprint + 1, std=aperture_size
                        ).reshape(2 * window_footprint + 1, 1)
    aperture = aperture * aperture.T
    grid_step_x = new_grid_x[1] - new_grid_x[0]
    grid_step_y = new_grid_y[1] - new_grid_y[0]
    subgrid_footprint = np.floor(
        (-1 + window_footprint * 0.5 / grid_step_x,
         -1 + window_footprint * 0.5 / grid_step_y))
    subgrid = (  # Add 2*(r_0 - r_M) to this to get s_desired
        window_footprint + 2 * grid_step_x * np.arange(
            -subgrid_footprint[0], subgrid_footprint[0] + 1),
        window_footprint + 2 * grid_step_y * np.arange(
            -subgrid_footprint[1], subgrid_footprint[1] + 1))
    subgrid_points = ((2 * subgrid_footprint[0] + 1) *
                      (2 * subgrid_footprint[1] + 1))
    for z in range(start_frame, end_frame + 1):
        im = load_image_slice(
            filename=data_filename, xPix=xPix, yPix=yPix,
            preframes=preframes, which_slice=z).astype(float)
        if hot_pixels is not None:
            im = remove_hot_pixels(im, hot_pixels)
        this_frames_enderlein_image.fill(0.)
        this_frames_normalization.fill(1e-12)
        if verbose:
            sys.stdout.write("\rProcessing raw data image %i" % (z))
            sys.stdout.flush()
        if make_widefield_image:
            widefield_image += interpolation.map_coordinates(im, widefield_coordinates).reshape(new_grid_y.shape[0],
                                                                                                new_grid_x.shape[0]).T
        lattice_points, i_list, j_list = (generate_lattice(image_shape=(xPix, yPix), lattice_vectors=lattice_vectors,
                                                           center_pix=offset_vector + get_shift(shift_vector, z),
                                                           edge_buffer=window_footprint + 1, return_i_j=True))
        for m, lp in enumerate(lattice_points):
            i, j = int(i_list[m]), int(j_list[m])
            """Take an image centered on each illumination point"""
            spot_image = get_centered_subimage(center_point=lp, window_size=window_footprint, image=im,
                                               background=background_frame)
            """Aperture the image with a synthetic pinhole"""
            intensity_normalization = 1.0 / (intensities_vs_galvo_position.get((i, j), {}).get(z, np.inf))
            if (intensity_normalization == 0 or spot_image.shape != (
                    2 * window_footprint + 1, 2 * window_footprint + 1)):
                continue  # Skip to the next spot
            apertured_image = (aperture * spot_image * intensity_normalization)
            nearest_grid_index = np.round((lp - (new_grid_x[0], new_grid_y[0])) / (grid_step_x, grid_step_y))
            nearest_grid_point = ((new_grid_x[0], new_grid_y[0]) + (grid_step_x, grid_step_y) * nearest_grid_index)
            new_coordinates = np.meshgrid(subgrid[0] + 2 * (nearest_grid_point[0] - lp[0]),
                                          subgrid[1] + 2 * (nearest_grid_point[1] - lp[1]))
            resampled_image = interpolation.map_coordinates(apertured_image, (
                new_coordinates[0].reshape(int(subgrid_points)),
                new_coordinates[1].reshape(int(subgrid_points)))).reshape(int(2 * subgrid_footprint[1] + 1),
                                                                          int(2 * subgrid_footprint[0] + 1)).T
            """Add the recentered image back to the scan grid"""
            if intensity_normalization > 0:
                this_frames_enderlein_image[
                int(nearest_grid_index[0] - subgrid_footprint[0]):int(nearest_grid_index[0] + subgrid_footprint[0] + 1),
                int(nearest_grid_index[1] - subgrid_footprint[1]):int(nearest_grid_index[1] + subgrid_footprint[1] + 1),
                ] += resampled_image
                this_frames_normalization[
                int(nearest_grid_index[0] - subgrid_footprint[0]):int(nearest_grid_index[0] + subgrid_footprint[0] + 1),
                int(nearest_grid_index[1] - subgrid_footprint[1]):int(nearest_grid_index[1] + subgrid_footprint[1] + 1),
                ] += 1
                if make_confocal_image:  # FIXME!!!!!!!
                    confocal_image[
                    nearest_grid_index[0] - window_footprint:nearest_grid_index[0] + window_footprint + 1,
                    nearest_grid_index[1] - window_footprint:nearest_grid_index[1] + window_footprint + 1
                    ] += interpolation.shift(
                        apertured_image, shift=(lp - nearest_grid_point))
            if show_steps:
                plt.clf()
                plt.suptitle(
                    "Spot %i, %i in frame %i\nCentered at %0.2f, %0.2f\n" % (i, j, z, lp[0], lp[1]) + (
                            "Nearest grid point: %i, %i" % (nearest_grid_point[0], nearest_grid_point[1])))
                plt.subplot(1, 3, 1)
                plt.imshow(
                    spot_image, interpolation='nearest', cmap="gray")
                plt.subplot(1, 3, 2)
                plt.imshow(apertured_image, interpolation='nearest', cmap="gray")
                plt.subplot(1, 3, 3)
                plt.imshow(resampled_image, interpolation='nearest', cmap="gray")
                fig.show()
                fig.canvas.draw()
                response = input('\nHit enter to continue, q to quit:')
                if response == 'q' or response == 'e' or response == 'x':
                    print("Done showing steps...")
                    show_steps = False
        enderlein_image += this_frames_enderlein_image
        enderlein_normalization += this_frames_normalization
        if not normalize:
            enderlein_normalization.fill(1)
            this_frames_normalization.fill(1)
        if intermediate_data:
            cumulative_sum[z, :, :] = (enderlein_image * 1. / enderlein_normalization)
            cumulative_sum.flush()
            processed_frames[z, :, :] = this_frames_enderlein_image * 1. / (this_frames_normalization)
            processed_frames.flush()
        if show_slices:
            plt.clf()
            plt.imshow(enderlein_image * 1.0 / enderlein_normalization, cmap="gray", interpolation='nearest')
            fig.show()
            fig.canvas.draw()
            response = input('Hit enter to continue...')

    images = {}
    images['enderlein_image'] = (enderlein_image * 1.0 / enderlein_normalization)
    if make_widefield_image:
        images['widefield_image'] = widefield_image
    if make_confocal_image:
        images['confocal_image'] = confocal_image
    return images


# def load_image_data(filename, xPix=512, yPix=512, zPix=201, preframes=0):
#     """Load the 16-bit raw data from the Visitech Infinity"""
#     return np.memmap(  # FIRST dimension is image number
#         filename, dtype=np.uint16, mode='r'
#     ).reshape(zPix + preframes, xPix, yPix)[preframes:, :, :]


def load_image_data(filename, xPix=512, yPix=512, zPix=201, preframes=0):
    """Load the 16-bit raw data from the Visitech Infinity"""
    _, image = cv2.imreadmulti(filename, flags=cv2.IMREAD_UNCHANGED)
    image = np.array(image)
    print(image.shape, image.dtype)

    mem_path = os.path.splitext(filename)[0]
    mem_path = mem_path + ".dat"

    if os.path.exists(mem_path):
        memmap_data = np.memmap(mem_path, dtype=image.dtype, mode='r', shape=image.size).reshape(zPix, xPix, yPix)
    else:
        memmap_data = np.memmap(mem_path, dtype=image.dtype, mode='w+', shape=image.size).reshape(zPix, xPix, yPix)
        memmap_data[:] = image[:]

    return memmap_data


def load_image_slice(filename, xPix, yPix, preframes=0, which_slice=0):
    """Load a frame of the 16-bit raw data from the Visitech Infinity"""
    bytes_per_pixel = 2

    _, image = cv2.imreadmulti(filename, flags=cv2.IMREAD_UNCHANGED)
    image = np.array(image)

    return image[which_slice, :, :]

    # mem_path = os.path.splitext(filename)[0]
    # mem_path = mem_path + ".dat"
    #
    # data_file = open(mem_path, 'rb')
    # data_file.seek((which_slice + preframes) * xPix * yPix * bytes_per_pixel)
    # try:
    #     return np.fromfile(
    #         data_file, dtype=np.uint16, count=xPix * yPix
    #     ).reshape(xPix, yPix)
    # except ValueError:
    #     print("\n\nWARNING: the data file:")
    #     print(data_file)
    #     print("may not be the size it was expected to be.\n\n")
    #     raise


def load_fft_slice(fft_data_folder, xPix, yPix, which_slice=0):
    bytes_per_pixel = 16
    filename = os.path.join(fft_data_folder, '%06i.dat' % (which_slice))
    data_file = open(filename, 'rb')
    return np.memmap(data_file, dtype=np.complex128, mode='r').reshape(xPix, yPix)


def get_fft_abs(filename, image_data, show_steps=False):
    basename = os.path.splitext(filename)[0]
    fft_abs_name = basename + '_fft_abs.npy'
    fft_avg_name = basename + '_fft_avg.npy'
    fft_data_folder = basename + '_fft_data'
    """FFT data is stored as a sequence of raw binary files, one per
    2D z-slice. The files are named 000000.dat, 000001.dat, etc."""

    if (os.path.exists(fft_abs_name) and
            os.path.exists(fft_avg_name) and
            os.path.exists(fft_data_folder)):
        print("Loading", os.path.split(fft_abs_name)[1])
        fft_abs = np.load(fft_abs_name)
        print("Loading", os.path.split(fft_avg_name)[1])
        fft_avg = np.load(fft_avg_name)
    else:
        print("Generating fft_abs, fft_avg and fft_data...")
        os.mkdir(fft_data_folder)
        fft_abs = np.zeros(image_data.shape[1:])
        fft_avg = np.zeros(image_data.shape[1:], dtype=np.complex128)
        window = (hann(image_data.shape[1]).reshape(image_data.shape[1], 1) *
                  hann(image_data.shape[2]).reshape(1, image_data.shape[2]))  # Multiplication of matrices
        if show_steps:
            fig = plt.figure()
        for z in range(image_data.shape[0]):
            fft_data = np.fft.fftshift(  # Stored shifted!
                np.fft.fftn(window * image_data[z, :, :], axes=(0, 1)))
            fft_data.tofile(os.path.join(fft_data_folder, '%06i.dat' % (z)))
            fft_abs += np.abs(fft_data)
            if show_steps:
                plt.clf()
                plt.subplot(1, 3, 1)
                plt.title('Windowed slice %i' % z)
                plt.imshow(window * np.array(image_data[z, :, :]), cmap="gray", interpolation='nearest')
                plt.subplot(1, 3, 2)
                plt.title('FFT of slice %i' % z)
                plt.imshow(np.log(1 + np.abs(fft_data)), cmap="gray", interpolation='nearest')
                plt.subplot(1, 3, 3)
                plt.title("Cumulative sum of FFT absolute values")
                plt.imshow(np.log(1 + fft_abs), cmap="gray", interpolation='nearest')
                plt.show()
                input("Hit enter to continue...")
            fft_avg += fft_data
            sys.stdout.write('\rFourier transforming slice %i' % (z + 1))
            sys.stdout.flush()
        fft_avg = np.abs(fft_avg)
        np.save(fft_abs_name, fft_abs)
        np.save(fft_avg_name, fft_avg)

    return fft_data_folder, fft_abs, fft_avg


def spike_filter(fft_abs, display=False):
    f = gaussian_filter(np.log(1 + fft_abs), sigma=0.5)
    if display:
        fig = plt.figure()
        plt.imshow(f, cmap="gray", interpolation='nearest')
        plt.title('Smoothed')
        fig.show()
        fig.canvas.draw()
        input('Hit enter...')
        plt.clf()
    f = f - gaussian_filter(f, sigma=(0, 4))
    if display:
        plt.imshow(f, cmap="gray", interpolation='nearest')
        plt.title('Filtered left-right')
        fig.show()
        fig.canvas.draw()
        input('Hit enter...')
        plt.clf()
    f = f - gaussian_filter(f, sigma=(4, 0))
    if display:
        plt.imshow(f, cmap="gray", interpolation='nearest')
        plt.title('Filtered up-down')
        fig.show()
        fig.canvas.draw()
        input('Hit enter...')
        plt.clf()
    f = gaussian_filter(f, sigma=(1.5))
    if display:
        plt.imshow(f, cmap="gray", interpolation='nearest')
        plt.title('Resmoothed')
        fig.show()
        fig.canvas.draw()
        input('Hit enter...')
        plt.clf()
    f = f * (f > 0)
    if display:
        plt.imshow(f, cmap="gray", interpolation='nearest')
        plt.title('Negative truncated')
        fig.show()
        fig.canvas.draw()
        input('Hit enter...')
        plt.clf()
    f -= f.mean()
    f *= 1.0 / f.std()
    return f


def find_spikes(fft_abs, filtered_fft_abs, extent=15, num_spikes=300,
                display=True, animate=False):
    """Finds spikes in the sum of the 2D ffts of an image stack"""
    center_pix = np.array(fft_abs.shape) // 2
    log_fft_abs = np.log(1 + fft_abs)
    filtered_fft_abs = np.array(filtered_fft_abs)

    if display:
        image_extent = [-0.5 - center_pix[1],
                        filtered_fft_abs.shape[1] - 0.5 - center_pix[1],
                        filtered_fft_abs.shape[0] - 0.5 - center_pix[0],
                        -0.5 - center_pix[0]]
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(log_fft_abs, cmap="gray", interpolation='nearest', extent=image_extent)
        plt.title('Average Fourier magnitude')
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(filtered_fft_abs), cmap="gray", interpolation='nearest', extent=image_extent)
        plt.title('Filtered average Fourier magnitude')
        plt.show()

    coords = []
    if animate:
        fig = plt.figure()
        print('Center pixel:', center_pix)
    for i in range(num_spikes):

        coords.append(np.array(np.unravel_index(filtered_fft_abs.argmax(), filtered_fft_abs.shape)))
        c = coords[-1]
        xSl = slice(max(c[0] - extent, 0), min(c[0] + extent, filtered_fft_abs.shape[0]))
        ySl = slice(max(c[1] - extent, 0), min(c[1] + extent, filtered_fft_abs.shape[1]))

        filtered_fft_abs[xSl, ySl] = 0

        if animate:
            print(i, ':', c)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(
                filtered_fft_abs, cmap="gray", interpolation='nearest')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.plot(filtered_fft_abs.max(axis=1))
            plt.show()
            if i == 0:
                input('.')

    coords = [c - center_pix for c in coords]
    coords = sorted(coords, key=lambda x: x[0] ** 2 + x[1] ** 2)

    return coords  # Lattice k-vectors, sorted by vector magnitude


def get_basis_vectors(fft_abs, coords, extent=15, tolerance=3., num_harmonics=3, verbose=False):
    for i in range(len(coords)):  # Where to start looking.
        basis_vectors = []
        precise_basis_vectors = []
        for c, coord in enumerate(coords):
            if c < i:
                continue

            if c == 0:
                if max(abs(coord)) > 0:
                    print("c:", c)
                    print("Coord:", coord)
                    print("Coordinates:")
                    for x in coords:
                        print(x)
                    raise UserWarning('No peak at the central pixel')
                else:
                    continue  # Don't look for harmonics of the DC term

            if coord[0] < 0 or (coord[0] == 0 and coord[1] < 0):
                # Ignore the negative versions
                if verbose:
                    print("\nIgnoring:", coord)
            else:
                # Check for harmonics
                if verbose:
                    print("\nTesting:", coord)
                num_vectors, points_found = test_basis(coords, [coord], tolerance=tolerance, verbose=verbose)
                if num_vectors > num_harmonics:
                    # We found enough harmonics. Keep it, for now.
                    basis_vectors.append(coord)
                    center_pix = np.array(fft_abs.shape) // 2
                    furthest_spike = points_found[-1] + center_pix
                    if verbose:
                        print("Appending", coord)
                        print("%i harmonics found, at:" % (num_vectors - 1))
                        for p in points_found:
                            print(' ', p)

                    if len(basis_vectors) > 1:
                        if verbose:
                            print("\nTesting combinations:", basis_vectors)
                        num_vectors, points_found = test_basis(coords, basis_vectors, tolerance=tolerance,
                                                               verbose=verbose)
                        if num_vectors > num_harmonics:
                            # The combination predicts the lattice
                            if len(basis_vectors) == 3:
                                # We're done; we have three consistent vectors.
                                precise_basis_vectors = get_precise_basis(coords, basis_vectors, fft_abs,
                                                                          tolerance=tolerance, verbose=verbose)
                                (x_1, x_2, x_3) = sorted(precise_basis_vectors, key=lambda x: abs(x[0]))
                                possibilities = sorted(
                                    ([x_1, x_2, x_3],
                                     [x_1, x_2, -x_3],
                                     [x_1, -x_2, x_3],
                                     [x_1, -x_2, -x_3]),
                                    key=lambda x: (np.array(sum(x)) ** 2).sum())
                                if verbose:
                                    print("Possible triangle combinations:")
                                    for p in possibilities:
                                        print(" ", p)
                                precise_basis_vectors = possibilities[0]
                                if precise_basis_vectors[-1][0] < 0:
                                    for p in range(3):
                                        precise_basis_vectors[p] *= -1
                                return precise_basis_vectors
                        else:
                            # Blame the new guy, for now.
                            basis_vectors.pop()
    else:
        raise UserWarning(
            "Basis vector search failed. Diagnose by running with verbose=True")


def test_basis(coords, basis_vectors, tolerance, verbose=False):
    # Checks for expected lattice, returns the points found and halts on failure.
    points_found = list(basis_vectors)
    num_vectors = 2
    searching = True
    while searching:
        if verbose:
            print("Looking for combinations of %i basis vectors." % num_vectors)
        lattice = [sum(c) for c in combinations_with_replacement(basis_vectors, num_vectors)]
        if verbose:
            print("Expected lattice points:", lattice)
        for i, lat in enumerate(lattice):
            for c in coords:
                dif = np.sqrt(((lat - c) ** 2).sum())
                if dif < tolerance:
                    if verbose:
                        print("Found lattice point:", c)
                        print(" Distance:", dif)
                        if len(basis_vectors) == 1:
                            print(" Fundamental:", c * 1.0 / num_vectors)
                    points_found.append(c)
                    break
            else:  # Fell through the loop
                if verbose:
                    print("Expected lattice point not found")
                searching = False
        if not searching:
            return num_vectors, points_found
        num_vectors += 1


def get_precise_basis(coords, basis_vectors, fft_abs, tolerance, verbose=False):
    # Uses the expected lattice to estimate precise values of the basis.
    if verbose:
        print("\nAdjusting basis vectors to match lattice...")
    center_pix = np.array(fft_abs.shape) // 2
    basis_vectors = list(basis_vectors)
    spike_indices = []
    spike_locations = []
    num_vectors = 2
    searching = True
    while searching:
        """I seem to be relying on combinations_with_replacemnet to
        give the same ordering twice in a row. Hope it always does!"""
        combinations = [
            c for c in combinations_with_replacement(basis_vectors,
                                                     num_vectors)]
        combination_indices = [
            c for c in combinations_with_replacement((0, 1, 2), num_vectors)]
        for i, comb in enumerate(combinations):
            lat = sum(comb)
            key = tuple([combination_indices[i].count(v) for v in (0, 1, 2)])
            for c in coords:
                dif = np.sqrt(((lat - c) ** 2).sum())
                if dif < tolerance:
                    p = c + center_pix
                    true_max = c + simple_max_finder(
                        fft_abs[p[0] - 1:p[0] + 2,
                        p[1] - 1:p[1] + 2], show_plots=False)
                    if verbose:
                        print("Found lattice point:", c)
                        print("Estimated position:", true_max)
                        print("Lattice index:", key)
                    spike_indices.append(key)
                    spike_locations.append(true_max)
                    break
            else:  # Fell through the loop
                if verbose:
                    print("Expected lattice point not found")
                searching = False
        if not searching:  # Given the spikes found, estimate the basis
            A = np.array(spike_indices)
            v = np.array(spike_locations)
            precise_basis_vectors, residues, rank, s = np.linalg.lstsq(A, v, rcond=None)
            if verbose:
                print("Precise basis vectors:")
                print(precise_basis_vectors)
                print("Residues:", residues)
                print("Rank:", rank)
                print("s:", s)
                print()
            return precise_basis_vectors
        num_vectors += 1


def combinations_with_replacement(iterable, r):
    """
    >>> print([i for i in combinations_with_replacement(['a', 'b', 'c'], 2)])
    [('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'b'), ('b', 'c'), ('c', 'c')]
    """
    pool = tuple(iterable)
    n = len(pool)
    for indices in product(range(n), repeat=r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)


def get_offset_vector(image, direct_lattice_vectors, prefilter='median', verbose=True, display=True,
                      show_interpolation=True):
    if prefilter == 'median':
        image = median_filter(image, size=3)

    if verbose:
        print("\nCalculating offset vector...")

    ws = 2 + int(max([abs(v).max() for v in direct_lattice_vectors]))
    if verbose:
        print("Window size:", ws)

    window = np.zeros([2 * ws + 1] * 2, dtype=np.float64)
    lattice_points = generate_lattice(image.shape, direct_lattice_vectors, edge_buffer=2 + ws)
    for lp in lattice_points:
        window += get_centered_subimage(center_point=lp, window_size=ws, image=image.astype(float))

    if display:
        plt.figure()
        plt.imshow(window, interpolation='nearest', cmap="gray")
        plt.title('Lattice average\nThis should look like round blobs')
        plt.show()

    buffered_window = np.array(window)
    buffered_window[:2, :] = 0
    buffered_window[-2:, :] = 0
    buffered_window[:, :2] = 0
    buffered_window[:, -2:] = 0

    while True:  # Don't want maxima on the edges
        max_pix = np.unravel_index(buffered_window.argmax(), window.shape)
        if (3 < max_pix[0] < window.shape[0] - 3) and (3 < max_pix[1] < window.shape[1] - 3):
            break
        else:
            buffered_window = gaussian_filter(buffered_window, sigma=2)

    if verbose:
        print("Maximum pixel in lattice average:", max_pix)

    correction = simple_max_finder(
        window[max_pix[0] - 1:max_pix[0] + 2, max_pix[1] - 1:max_pix[1] + 2],
        show_plots=show_interpolation)

    offset_vector = max_pix + correction + np.array(image.shape) // 2 - ws
    if verbose:
        print("Offset vector:", offset_vector)

    return offset_vector


def simple_max_finder(a, show_plots=True):
    """Given a 3x3 array with the maximum pixel in the center,
    estimates the x/y position of the true maximum"""
    true_max = []
    inter_points = np.arange(-1, 2)
    for data in (a[:, 1], a[1, :]):
        my_fit = np.poly1d(np.polyfit(inter_points, data, deg=2))
        true_max.append(-my_fit[1] / (2.0 * my_fit[2]))

    true_max = np.array(true_max)

    if show_plots:
        print("Correction:", true_max)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(a, interpolation='nearest', cmap="gray")
        plt.axhline(y=1 + true_max[0])
        plt.axvline(x=1 + true_max[1])
        plt.subplot(1, 3, 2)
        plt.plot(a[:, 1])
        plt.axvline(x=1 + true_max[0])
        plt.subplot(1, 3, 3)
        plt.plot(a[1, :])
        plt.axvline(x=1 + true_max[1])
        plt.show()

    return true_max


def get_shift_vector(
        fourier_lattice_vectors, fft_data_folder, filtered_fft_abs, num_harmonics=3, outlier_phase=1., verbose=True,
        display=True, scan_type='visitech', scan_dimensions=None):
    if verbose:
        print("\nCalculating shift vector...")

    center_pix = np.array(filtered_fft_abs.shape) // 2
    harmonic_pixels = []
    values = {}
    for v in fourier_lattice_vectors:
        harmonic_pixels.append([])
        for i in range(1, num_harmonics + 1):
            expected_pix = (np.round((i * v)) + center_pix).astype(int)
            roi = filtered_fft_abs[expected_pix[0] - 1:expected_pix[0] + 2, expected_pix[1] - 1:expected_pix[1] + 2]
            shift = -1 + np.array(
                np.unravel_index(roi.argmax(), roi.shape))
            actual_pix = expected_pix + shift - center_pix
            if verbose:
                print("Expected pixel:", expected_pix - center_pix)
                print("Shift:", shift)
                print("Brightest neighboring pixel:", actual_pix)
            harmonic_pixels[-1].append(tuple(actual_pix))
            values[harmonic_pixels[-1][-1]] = []

    num_slices = len(os.listdir(fft_data_folder))
    if verbose:
        print('\n')

    for z in range(num_slices):
        if verbose:
            sys.stdout.write("\rLoading harmonic pixels from FFT slice %06i" % z)
            sys.stdout.flush()
        fft_data = load_fft_slice(fft_data_folder, xPix=filtered_fft_abs.shape[0], yPix=filtered_fft_abs.shape[1],
                                  which_slice=z)
        for hp in harmonic_pixels:
            for p in hp:
                values[p].append(fft_data[p[0] + center_pix[0], p[1] + center_pix[1]])

    if verbose:
        print()

    slopes = []
    k = []
    if display:
        plt.figure()
    if scan_dimensions is not None:
        scan_dimensions = tuple(reversed(scan_dimensions))
    for hp in harmonic_pixels:
        for n, p in enumerate(hp):
            values[p] = np.unwrap(np.angle(values[p]))
            if scan_type == 'visitech':
                slope = np.polyfit(range(len(values[p])), values[p], deg=1)[0]
                values[p] -= slope * np.arange(len(values[p]))
            elif scan_type == 'dmd':
                if scan_dimensions[0] * scan_dimensions[1] != num_slices:
                    raise UserWarning(
                        "The scan dimensions are %i by %i," +
                        " but there are %i slices" % (scan_dimensions[0], scan_dimensions[1], num_slices))
                slope = [0, 0]
                slope[0] = np.polyfit(range(scan_dimensions[1]),
                                      values[p].reshape(scan_dimensions).sum(axis=0) * 1.0 / scan_dimensions[0],
                                      deg=1)[0]
                values[p] -= slope[0] * np.arange(len(values[p]))
                slope[1] = np.polyfit(
                    scan_dimensions[1] * np.arange(scan_dimensions[0]),
                    values[p].reshape(
                        scan_dimensions).sum(axis=1) * 1.0 / scan_dimensions[1],
                    deg=1)[0]
                values[p] -= slope[1] * scan_dimensions[1] * (np.arange(len(values[p])) // scan_dimensions[1])
                slope[1] *= scan_dimensions[1]
            values[p] -= values[p].mean()
            if abs(values[p]).mean() < outlier_phase:
                k.append(p * (-2. * np.pi / np.array(fft_data.shape)))
                slopes.append(slope)
            else:
                if verbose:
                    print("Ignoring outlier:", p)
            if display:
                plt.plot(values[p], '.-', label=repr(p))
    if display:
        plt.title('This should look like noise. Sudden jumps mean bad data!')
        plt.ylabel('Deviation from expected phase')
        plt.xlabel('Image number')
        plt.grid()
        plt.legend(prop={'size': 8})
        plt.axis('tight')
        x_limits = 1.05 * np.array(plt.xlim())
        x_limits -= x_limits[-1] * 0.025
        plt.xlim(x_limits)
        plt.show()

    if scan_type == 'visitech':
        x_s, residues, rank, s = np.linalg.lstsq(np.array(k), np.array(slopes), rcond=None)
    elif scan_type == 'dmd':
        x_s, residues, rank, s = {}, [0, 0], [0, 0], [0, 0]
        x_s['fast_axis'], residues[0], rank[0], s[0] = np.linalg.lstsq(np.array(k), np.array([sl[0] for sl in slopes]),
                                                                       rcond=None)
        x_s['slow_axis'], residues[1], rank[1], s[1] = np.linalg.lstsq(np.array(k), np.array([sl[1] for sl in slopes]),
                                                                       rcond=None)
        x_s['scan_dimensions'] = tuple(reversed(scan_dimensions))

    if verbose:
        print("Shift vector:")
        pprint.pprint(x_s)
        print("Residues:", residues)
        print("Rank:", rank)
        print("s:", s)
    return x_s


def get_precise_shift_vector(
        direct_lattice_vectors, shift_vector, offset_vector, last_image, zPix, scan_type, verbose):
    """Use the offset vector to correct the shift vector"""
    final_offset_vector = get_offset_vector(
        image=last_image,
        direct_lattice_vectors=direct_lattice_vectors,
        verbose=False, display=False, show_interpolation=False)

    final_lattice = generate_lattice(last_image.shape, direct_lattice_vectors,
                                     center_pix=offset_vector + get_shift(shift_vector, zPix - 1))
    closest_approach = 1e12
    print(type(last_image.shape[0]))
    for p in final_lattice:
        dif = p - final_offset_vector
        distance_sq = (dif ** 2).sum()
        if distance_sq < closest_approach:
            closest_lattice_point = p
            closest_approach = distance_sq
    shift_error = closest_lattice_point - final_offset_vector
    if scan_type == 'visitech':
        movements = zPix - 1
        corrected_shift_vector = shift_vector - (shift_error * 1.0 / movements)
    elif scan_type == 'dmd':
        movements = ((zPix - 1) // shift_vector['scan_dimensions'][0])
        corrected_shift_vector = dict(shift_vector)
        corrected_shift_vector['slow_axis'] = (shift_vector['slow_axis'] - shift_error * 1.0 / movements)

    if verbose:
        print("\nCorrecting shift vector...")
        print(" Initial shift vector:")
        print(' ', pprint.pprint(shift_vector))
        print(" Final offset vector:", final_offset_vector)
        print(" Closest predicted lattice point:", closest_lattice_point)
        print(" Error:", shift_error, "in", movements, "movements")
        print(" Corrected shift vector:")
        print(' ', pprint.pprint(corrected_shift_vector))
        print()

    return corrected_shift_vector, final_offset_vector


def get_shift(shift_vector, frame_number):
    if isinstance(shift_vector, dict):
        """This means we have a 2D shift vector"""
        fast_steps = frame_number % shift_vector['scan_dimensions'][0]
        slow_steps = frame_number // shift_vector['scan_dimensions'][0]
        return (shift_vector['fast_axis'] * fast_steps +
                shift_vector['slow_axis'] * slow_steps)
    else:
        """This means we have a 1D shift vector, like the Visitech Infinity"""
        return frame_number * shift_vector


def show_lattice_overlay(
        image_data, direct_lattice_vectors, offset_vector, shift_vector):
    plt.figure()
    s = 0
    while True:
        plt.clf()
        show_me = median_filter(np.array(image_data[s, :, :]), size=3)
        dots = np.zeros(list(show_me.shape) + [4])
        lattice_points = generate_lattice(
            show_me.shape, direct_lattice_vectors,
            center_pix=offset_vector + get_shift(shift_vector, s))
        for lp in lattice_points:
            x, y = np.round(lp).astype(int)
            dots[x, y, 0::3] = 1
        plt.imshow(show_me, cmap="gray", interpolation='nearest')
        plt.imshow(dots, interpolation='nearest')
        plt.title("Red dots show the calculated illumination pattern")
        plt.show()

        new_s = input("Next frame [exit]:")
        if new_s == '':
            print("Exiting")
            break
        try:
            s = int(new_s)
        except ValueError:
            print("Response not understood. Exiting display.")
            break
        s %= image_data.shape[0]
        print("Displaying frame %i" % (s))

    return None


def show_illuminated_points(
        direct_lattice_vectors, shift_vector, offset_vector='image', xPix=120, yPix=120, step_size=1, num_steps=200,
        verbose=True):
    if verbose:
        print("\nShowing a portion of the illumination points...")

    spots = sum(combine_lattices(direct_lattice_vectors, shift_vector, offset_vector, xPix, yPix, step_size, num_steps,
                                 verbose=verbose), [])
    fig = plt.figure()
    plt.plot([p[1] for p in spots], [p[0] for p in spots], '.')
    plt.xticks(range(yPix))
    plt.yticks(range(xPix))
    plt.grid()
    plt.axis('equal')
    plt.show()

    return fig


def combine_lattices(
        direct_lattice_vectors, shift_vector, offset_vector='image', xPix=120, yPix=120, step_size=1, num_steps=200,
        edge_buffer=2, verbose=True):
    if verbose:
        print("Combining lattices...")

    if offset_vector == 'image':
        offset_vector = np.array((xPix // 2, yPix // 2))

    spots = []
    for i in range(num_steps):
        spots.append([])
        if verbose:
            sys.stdout.write('\rz: %04i' % (i + 1))
            sys.stdout.flush()
        spots[-1] += generate_lattice(
            image_shape=(xPix, yPix),
            lattice_vectors=direct_lattice_vectors,
            center_pix=offset_vector + get_shift(shift_vector, i * step_size),
            edge_buffer=edge_buffer)

    if verbose:
        print()

    return spots


def spot_intensity_vs_galvo_position(
        lake_filename, xPix, yPix, zPix, preframes, direct_lattice_vectors, shift_vector, offset_vector,
        background_filename, background_zPix, window_size=5, verbose=False, show_steps=False, display=False):
    """Calibrate how the intensity of each spot varies with galvo
    position, using a fluorescent lake dataset and a stack of
    light-free background images."""

    lake_basename = os.path.splitext(lake_filename)[0]
    lake_intensities_name = lake_basename + '_spot_intensities.pkl'
    background_basename = os.path.splitext(background_filename)[0]
    background_name = background_basename + '_background_image.raw'
    background_directory_name = os.path.dirname(background_basename)

    try:
        hot_pixels = np.fromfile(
            os.path.join(background_directory_name, 'hot_pixels.txt'), sep=', ')
    except IOError:
        skip_hot_pix = input("Hot pixel list not found. Continue? [y]/n:")
        if skip_hot_pix == 'n':
            raise
        else:
            hot_pixels = None
    else:
        hot_pixels = hot_pixels.reshape(2, len(hot_pixels) // 2)

    if os.path.exists(lake_intensities_name) and os.path.exists(background_name):
        print("\nIllumination intensity calibration already calculated.")
        print("Loading", os.path.split(lake_intensities_name)[1])
        intensities_vs_galvo_position = pickle.load(open(lake_intensities_name, 'rb'))
        print("Loading", os.path.split(background_name)[1])
        try:
            bg = np.fromfile(background_name, dtype=float).reshape(xPix, yPix)
        except ValueError:
            print("\n\nWARNING: the data file:")
            print(background_name)
            print("may not be the size it was expected to be.\n\n")
            raise
    else:
        print("\nCalculating illumination spot intensities...")
        print("Constructing background image...")
        background_image_data = load_image_data(background_filename, xPix, yPix, background_zPix, preframes)
        bg = np.zeros((xPix, yPix), dtype=float)
        for z in range(background_image_data.shape[0]):
            bg += background_image_data[z, :, :]
        bg *= 1.0 / background_image_data.shape[0]
        del background_image_data
        if hot_pixels is not None:
            bg = remove_hot_pixels(bg, hot_pixels)
        print("Background image complete.")

        lake_image_data = load_image_data(lake_filename, xPix, yPix, zPix, preframes)
        intensities_vs_galvo_position = {}
        """A dict of dicts. Element [i, j][z] gives the intensity of the
        i'th, j'th spot in the lattice, in frame z"""
        if show_steps:
            plt.figure()
        print("Computing flat-field calibration...")
        for z in range(lake_image_data.shape[0]):
            im = np.array(lake_image_data[z, :, :], dtype=float)
            if hot_pixels is not None:
                im = remove_hot_pixels(im, hot_pixels)
            sys.stdout.write("\rCalibration image %i" % z)
            sys.stdout.flush()
            lattice_points, i_list, j_list = generate_lattice(
                image_shape=(xPix, yPix),
                lattice_vectors=direct_lattice_vectors,
                center_pix=offset_vector + get_shift(shift_vector, z),
                edge_buffer=window_size + 1,
                return_i_j=True)

            for m, lp in enumerate(lattice_points):
                i, j = int(i_list[m]), int(j_list[m])
                intensity_history = intensities_vs_galvo_position.setdefault((i, j), {})  # Get this spot's history
                spot_image = get_centered_subimage(
                    center_point=lp, window_size=window_size,
                    image=im, background=bg)
                intensity_history[z] = float(spot_image.sum())  # Funny thing...
                if show_steps:
                    plt.clf()
                    plt.imshow(spot_image, interpolation='nearest', cmap="gray")
                    plt.title("Spot %i, %i in frame %i\nCentered at %0.2f, %0.2f" % (i, j, z, lp[0], lp[1]))
                    plt.show()
                    response = input()
                    if response == 'q' or response == 'e' or response == 'x':
                        print("Done showing steps...")
                        show_steps = False

        """Normalize the intensity values"""
        num_entries = 0
        total_sum = 0
        for hist in intensities_vs_galvo_position.values():
            for intensity in hist.values():
                num_entries += 1
                total_sum += intensity
        inverse_avg = num_entries * 1.0 / total_sum
        for hist in intensities_vs_galvo_position.values():
            for k in hist.keys():
                hist[k] *= inverse_avg
        print("\nSaving", os.path.split(lake_intensities_name)[1])
        pickle.dump(intensities_vs_galvo_position, open(lake_intensities_name, 'wb'), protocol=2)
        print("Saving", os.path.split(background_name)[1])
        bg.tofile(background_name)

    if display:
        plt.figure()
        num_lines = 0
        for (i, j), spot_hist in intensities_vs_galvo_position.items()[:10]:
            num_lines += 1
            sh = spot_hist.items()
            plt.plot([frame_num for frame_num, junk in sh],
                     [intensity for junk, intensity in sh],
                     ('-', '-.')[num_lines > 5],
                     label=repr((i, j)))
        plt.legend()
        plt.show()
    return intensities_vs_galvo_position, bg  # bg is short for 'background'


def remove_hot_pixels(image, hot_pixels):
    for y, x in hot_pixels:
        image[int(x), int(y)] = np.median(image[int(x) - 1:int(x) + 2, int(y) - 1:int(y) + 2])
    return image


def generate_lattice(image_shape, lattice_vectors, center_pix='image', edge_buffer=2, return_i_j=False):
    # 修正判断
    if isinstance(center_pix, str):
        if center_pix == 'image':
            center_pix = np.array(image_shape) // 2
    else:
        center_pix = np.array(center_pix) - (np.array(image_shape) // 2)
        lattice_components = np.linalg.solve(np.vstack(lattice_vectors[:2]).T, center_pix)
        lattice_components_centered = np.mod(lattice_components, 1)
        lattice_shift = lattice_components - lattice_components_centered
        center_pix = (lattice_vectors[0] * lattice_components_centered[0] +
                      lattice_vectors[1] * lattice_components_centered[1] +
                      np.array(image_shape) // 2)

    num_vectors = int(np.round(1.5 * max(image_shape) / np.sqrt((lattice_vectors[0] ** 2).sum())))  # changed
    lower_bounds = (edge_buffer, edge_buffer)
    upper_bounds = (image_shape[0] - edge_buffer, image_shape[1] - edge_buffer)
    i, j = np.mgrid[-num_vectors:num_vectors, -num_vectors:num_vectors]
    i = i.reshape(i.size, 1)
    j = j.reshape(j.size, 1)
    lp = i * lattice_vectors[0] + j * lattice_vectors[1] + center_pix
    valid = np.all(lower_bounds < lp, 1) * np.all(lp < upper_bounds, 1)
    lattice_points = list(lp[valid])
    if return_i_j:
        return (lattice_points,
                list(i[valid] - lattice_shift[0]),
                list(j[valid] - lattice_shift[1]))
    else:
        return lattice_points


def get_centered_subimage(
        center_point, window_size, image, background='none'):
    x, y = np.round(center_point).astype(int)
    xSl = slice(max(x - window_size - 1, 0), x + window_size + 2)
    ySl = slice(max(y - window_size - 1, 0), y + window_size + 2)
    subimage = np.array(image[xSl, ySl])

    if not isinstance(background, str):
        subimage -= background[xSl, ySl]
    interpolation.shift(subimage, shift=(x, y) - center_point, output=subimage)
    return subimage[1:-1, 1:-1]


def join_enderlein_images(
        data_filenames_list, new_grid_xrange, new_grid_yrange, join_widefield_images=True):
    if len(data_filenames_list) < 2:
        print("Less than two files to join. Skipping...")
        return None

    print("Joining enderlein and widefield images into stack...")

    enderlein_stack = np.zeros((len(data_filenames_list), new_grid_xrange[2], new_grid_yrange[2]), dtype=np.float64)
    if join_widefield_images:
        widefield_stack = np.zeros((len(data_filenames_list), new_grid_xrange[2], new_grid_yrange[2]), dtype=np.float64)

    for i, d in enumerate(data_filenames_list):
        sys.stdout.write('\rLoading file %i of %i' % (i, len(data_filenames_list)))
        sys.stdout.flush()

        basename = os.path.splitext(d)[0]
        enderlein_image_name = basename + '_enderlein_image.raw'
        widefield_image_name = basename + '_widefield.raw'

        try:
            enderlein_stack[i, :, :] = np.fromfile(enderlein_image_name, dtype=np.float64).reshape(new_grid_xrange[2],
                                                                                                   new_grid_yrange[2])

        except ValueError:
            print("\n\nWARNING: the data file:")
            print(enderlein_image_name)
            print("may not be the size it was expected to be.\n\n")
            raise

        if join_widefield_images:
            try:
                widefield_stack[i, :, :] = np.fromfile(
                    widefield_image_name, dtype=np.float64).reshape(
                    new_grid_xrange[2], new_grid_yrange[2])
            except ValueError:
                print("\n\nWARNING: the data file:")
                print(widefield_image_name)
                print("may not be the size it was expected to be.\n\n")
                raise

    stack_basename = os.path.commonprefix(data_filenames_list).rstrip('0123456789')
    print("\nStack basename:", stack_basename)

    enderlein_stack.tofile(stack_basename + '_enderlein_stack.raw')
    if join_widefield_images:
        widefield_stack.tofile(stack_basename + '_widefield_stack.raw')

        with open(stack_basename + '_widefield_stack.txt', 'w') as w_notes:
            w_notes.write("Left/right: %i pixels\n" % (widefield_stack.shape[2]))
            w_notes.write("Up/down: %i pixels\n" % (widefield_stack.shape[1]))
            w_notes.write("Number of images: %i\n" % (widefield_stack.shape[0]))
            w_notes.write("Data type: 64-bit real\n")
            w_notes.write("Byte order: Intel (little-endian)\n")

        # w_notes = open(stack_basename + '_widefield_stack.txt', 'wb')
        # w_notes.write("Left/right: %i pixels\r\n" % (widefield_stack.shape[2]))
        # w_notes.write("Up/down: %i pixels\r\n" % (widefield_stack.shape[1]))
        # w_notes.write("Number of images: %i\r\n" % (widefield_stack.shape[0]))
        # w_notes.write("Data type: 64-bit real\r\n")
        # w_notes.write("Byte order: Intel (little-endian))\r\n")
        # w_notes.close()

    with open(stack_basename + '_enderlein_stack.txt', 'w') as e_notes:
        e_notes.write("Left/right: %i pixels\n" % (enderlein_stack.shape[2]))
        e_notes.write("Up/down: %i pixels\n" % (enderlein_stack.shape[1]))
        e_notes.write("Number of images: %i\n" % (enderlein_stack.shape[0]))
        e_notes.write("Data type: 64-bit real\n")
        e_notes.write("Byte order: Intel (little-endian)\n")

    # e_notes = open(stack_basename + '_enderlein_stack.txt', 'wb')
    # e_notes.write("Left/right: %i pixels\r\n" % (enderlein_stack.shape[2]))
    # e_notes.write("Up/down: %i pixels\r\n" % (enderlein_stack.shape[1]))
    # e_notes.write("Number of images: %i\r\n" % (enderlein_stack.shape[0]))
    # e_notes.write("Data type: 64-bit real\r\n")
    # e_notes.write("Byte order: Intel (little-endian))\r\n")
    # e_notes.close()

    print("Done joining.")
    return None


def get_data_locations():
    """Assumes that hot_pixels.txt and background.raw are in the same
    directory ast array_illumination.py"""
    import tkinter as tk
    import tkinter.filedialog as tkFileDialog
    import tkinter.simpledialog as tkSimpleDialog
    import glob, array_illumination

    module_dir = os.path.dirname(array_illumination.__file__)
    background_filename = os.path.join(module_dir, 'background.raw')

    tkroot = tk.Tk()
    tkroot.withdraw()

    data_filename = str(os.path.normpath(tkFileDialog.askopenfilename(
        title="Select one of your raw SIM data files",
        filetypes=[('Raw binary', '.raw')],
        defaultextension='.raw',
        initialdir=os.getcwd()
    )))  # Careful about Unicode here!
    data_dir = os.path.dirname(data_filename)

    while True:
        wildcard_data_filename = tkSimpleDialog.askstring(
            title='Filename pattern',
            prompt=("Use '?' as a wildcard\n\n" +
                    "For example:\n" +
                    "  image_????.raw\n" +
                    "would match:\n" +
                    "  image_0001.raw\n" +
                    "  image_0002.raw\n" +
                    "  etc...\n" +
                    "but would not match:\n" +
                    "   image_001.raw"),
            initialvalue=os.path.split(data_filename)[1])

        data_filenames_list = sorted(glob.glob(os.path.join(data_dir, wildcard_data_filename)))
        print("Data filenames:")
        for f in data_filenames_list:
            print('.  ' + f)

        response = input("Are those the files you want to process? [y]/n:")
        if response == 'n':
            continue
        else:
            break

    lake_filename = str(os.path.normpath(tkFileDialog.askopenfilename(
        title="Select your lake calibration raw data file",
        filetypes=[('Raw binary', '.raw')],
        defaultextension='.raw',
        initialdir=os.path.join(data_dir, os.pardir),
        initialfile='lake.raw'
    )))  # Careful about Unicode here!

    tkroot.destroy()
    return data_dir, data_filenames_list, lake_filename, background_filename


def use_lake_parameters():
    import tkinter as tk
    import tkinter.messagebox as tkMessageBox

    tkroot = tk.Tk()
    tkroot.withdraw()
    use_all_lake_parameters = tkMessageBox.askyesno(
        default=tkMessageBox.NO,
        icon=tkMessageBox.QUESTION,
        message="Use lake to determine offset?\n(Useful for sparse samples)",
        title='Offset calculation')
    tkroot.destroy()
    return use_all_lake_parameters


if __name__ == '__main__':
    print(get_data_locations())
