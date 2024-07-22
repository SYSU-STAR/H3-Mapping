#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ELSED.h>

namespace py = pybind11;
using namespace upm;

// Converts C++ descriptors to Numpy
inline py::tuple salient_segments_to_py(const upm::SalientSegments &ssegs) {
  py::array_t<float> scores(ssegs.size());
  py::array_t<float> segments({int(ssegs.size()), 4});
  float *p_scores = scores.mutable_data();
  float *p_segments = segments.mutable_data();
  for (int i = 0; i < ssegs.size(); i++) {
    p_scores[i] = ssegs[i].salience;
//    std::cout << "salience: " << ssegs[i].salience << std::endl;
    p_segments[i * 4] = ssegs[i].segment[0];
    p_segments[i * 4 + 1] = ssegs[i].segment[1];
    p_segments[i * 4 + 2] = ssegs[i].segment[2];
    p_segments[i * 4 + 3] = ssegs[i].segment[3];
  }
  return pybind11::make_tuple(segments, scores);
}

py::tuple compute_elsed(const py::array &py_img,
                        float sigma = 1,
                        float gradientThreshold = 30,
                        int minLineLen = 15,
                        double lineFitErrThreshold = 0.2,
                        double pxToSegmentDistTh = 1.5,
                        double validationTh = 0.15,
                        bool validate = true,
                        bool treatJunctions = true
) {

  py::buffer_info info = py_img.request();
  cv::Mat img(info.shape[0], info.shape[1], CV_8UC1, (uint8_t *) info.ptr);
  ELSEDParams params;


  params.sigma = sigma;
  params.ksize = cvRound(sigma * 3 * 2 + 1) | 1; // Automatic kernel size detection
  params.gradientThreshold = gradientThreshold;
  params.minLineLen = minLineLen;
  params.lineFitErrThreshold = lineFitErrThreshold;
  params.pxToSegmentDistTh = pxToSegmentDistTh;
  params.validationTh = validationTh;
  params.validate = validate;
  params.treatJunctions = treatJunctions;

  ELSED elsed(params);
  upm::SalientSegments salient_segs = elsed.detectSalient(img);
  return salient_segments_to_py(salient_segs);
}

py::tuple compute_elsed_consider_pc_dist(const py::array &py_img,
                        const py::array &py_pc_dist,
                        float sigma = 1,
                        float ksize = 5,
                        float gradientThreshold = 30,
                        int minLineLen = 15,
                        double lineFitErrThreshold = 0.2,
                        double pxToSegmentDistTh = 1.5,
                        double validationTh = 0.15,
                        bool validate = true,
                        bool treatJunctions = true,
                        float pc_dist_threshold = 0.01
                                ) {

    py::buffer_info info = py_img.request();
    cv::Mat img(info.shape[0], info.shape[1], CV_8UC1, (uint8_t *) info.ptr);
    py::buffer_info info_pc_dist = py_pc_dist.request();
    cv::Mat pc_dist(info_pc_dist.shape[0], info_pc_dist.shape[1], CV_32FC1, (float *) info_pc_dist.ptr);
    ELSEDParams params;

    params.sigma = sigma;
    params.ksize = ksize;
//    params.ksize = cvRound(sigma * 3 * 2 + 1) | 1; // Automatic kernel size detection
    params.gradientThreshold = gradientThreshold;
    params.minLineLen = minLineLen;
    params.lineFitErrThreshold = lineFitErrThreshold;
    params.pxToSegmentDistTh = pxToSegmentDistTh;
    params.validationTh = validationTh;
    params.validate = validate;
    params.treatJunctions = treatJunctions;
    params.pc_dist_threshold = pc_dist_threshold;

    ELSED elsed(params);
    upm::SalientSegments salient_segs = elsed.detectSalient_pc_dist(img, pc_dist);

    return salient_segments_to_py(salient_segs);
}

PYBIND11_MODULE(pyelsed, m) {
  m.def("detect", &compute_elsed, R"pbdoc(
        Computes ELSED: Enhanced Line SEgment Drawing in the input image.
    )pbdoc",
        py::arg("img"),
        py::arg("sigma") = 1,
        py::arg("gradientThreshold") = 30,
        py::arg("minLineLen") = 15,
        py::arg("lineFitErrThreshold") = 0.2,
        py::arg("pxToSegmentDistTh") = 1.5,
        py::arg("validationTh") = 0.15,
        py::arg("validate") = true,
        py::arg("treatJunctions") = true
  );
  m.def("detect_consider_pc_dist", &compute_elsed_consider_pc_dist, R"pbdoc(
        Computes ELSED consider pc_dist: Enhanced Line SEgment Drawing in the input image.
  )pbdoc",
  py::arg("img"),
  py::arg("pc_dist"),
  py::arg("sigma") = 1,
  py::arg("ksize") = 5,
  py::arg("gradientThreshold") = 30,
  py::arg("minLineLen") = 15,
  py::arg("lineFitErrThreshold") = 0.2,
  py::arg("pxToSegmentDistTh") = 1.5,
  py::arg("validationTh") = 0.15,
  py::arg("validate") = true,
  py::arg("treatJunctions") = true,
  py::arg("pc_dist_threshold") = 0.01
  );
}
