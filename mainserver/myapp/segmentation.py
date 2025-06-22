
import numpy as np

class SegmentationAnalyzer:
    def __init__(self, masks, classes, class_names, scale_bar_microns=0.5, scale_bar_pixels=72):
        self.masks = masks
        self.classes = classes
        self.class_names = class_names
        self.scale_bar_microns = scale_bar_microns
        self.scale_bar_pixels = scale_bar_pixels
        self.micron_per_pixel = scale_bar_microns / scale_bar_pixels

    def _polygon_area(self, polygon):
        x = polygon[:, 0]
        y = polygon[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _polygon_perimeter(self, polygon):
        return np.sum(np.sqrt(np.sum(np.diff(np.vstack([polygon, polygon[0]]), axis=0)**2, axis=1)))

    def _pixel_to_micron2(self, pixel_area):
        return pixel_area * (self.micron_per_pixel ** 2)

    def _pixel_to_micron(self, pixel_length):
        return pixel_length * self.micron_per_pixel

    def analyze(self):
        results = {
            "Etioplast": {"area_um2": 0.0, "count": 0},
            "PLB": {"area_um2": 0.0, "count": 0},
            "Prothylakoid": {"count": 0, "total_length_um": 0.0},
            "Plastoglobule": {"count": 0, "diameters_um": []}
        }

        for seg, cls_idx in zip(self.masks, self.classes):
            label = self.class_names[cls_idx]
            polygon = np.array(seg, dtype=np.float32)

            if label == "Etioplast":
                area_px = self._polygon_area(polygon)
                area_um2 = self._pixel_to_micron2(area_px)
                results[label]["area_um2"] += area_um2
                results[label]["count"] += 1

            elif label == "PLB":
                area_px = self._polygon_area(polygon)
                area_um2 = self._pixel_to_micron2(area_px)
                results[label]["area_um2"] += area_um2
                results[label]["count"] += 1

            elif label == "Prothylakoid":
                perimeter_px = self._polygon_perimeter(polygon)
                length_um = self._pixel_to_micron(perimeter_px)
                results[label]["total_length_um"] += length_um
                results[label]["count"] += 1

            elif label == "Plastoglobule":
                area_px = self._polygon_area(polygon)
                area_um2 = self._pixel_to_micron2(area_px)
                diameter_um = 2 * np.sqrt(area_um2 / np.pi)  # assuming circular shape
                results[label]["diameters_um"].append(diameter_um)
                results[label]["count"] += 1

        # # Round results
        # results["Etioplast"]["area_um2"] = round(results["Etioplast"]["area_um2"], 2)
        # results["PLB"]["area_um2"] = round(results["PLB"]["area_um2"], 2)
        # results["Prothylakoid"]["total_length_um"] = round(results["Prothylakoid"]["total_length_um"], 2)
        # if results["Plastoglobule"]["diameters_um"]:
        #     avg_diameter = np.mean(results["Plastoglobule"]["diameters_um"])
        #     results["Plastoglobule"]["avg_diameter_um"] = round(avg_diameter, 2)
        # else:
        #     results["Plastoglobule"]["avg_diameter_um"] = 0.0

        # del results["Plastoglobule"]["diameters_um"]  # remove raw list before returning

        results["Etioplast"]["area_um2"] = float(f"{results['Etioplast']['area_um2']:.2f}")
        results["PLB"]["area_um2"] = float(f"{results['PLB']['area_um2']:.2f}")
        results["Prothylakoid"]["total_length_um"] = float(f"{results['Prothylakoid']['total_length_um']:.2f}")

        if results["Plastoglobule"]["diameters_um"]:
            avg_diameter = np.mean(results["Plastoglobule"]["diameters_um"])
            results["Plastoglobule"]["avg_diameter_um"] = float(f"{avg_diameter:.2f}")
        else:
            results["Plastoglobule"]["avg_diameter_um"] = 0.0

        del results["Plastoglobule"]["diameters_um"]

        return results, round(self.micron_per_pixel, 6)
