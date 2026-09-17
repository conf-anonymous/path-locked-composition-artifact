// Calls the original KITTI devkit's read-only geometry functions, never its
// main/eval/plot/mail routines. Inputs are actual recorded training poses.
#include <string>
// The shipped evaluator calls Mail::finalize, absent from its shipped mail.h.
// Supply a no-I/O adapter only for the unused server wrapper. Geometry source
// stays byte-identical, and this probe cannot send notification emails.
#define MAIL_H
class Mail {
 public:
  explicit Mail(std::string = "") {}
  void msg(const char *, ...) {}
  template<class... Args> void finalize(Args...) {}
};
#define main unused_official_main
#include "evaluate_odometry.cpp"
#undef main
#include <iomanip>

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  auto poses = loadPoses(argv[1]);
  if (poses.empty()) return 3;
  auto dist = trajectoryDistances(poses);
  std::cout << std::setprecision(17);
  for (int i=0; i<dist.size(); ++i)
    std::cout << "D " << i << " " << dist[i] << "\n";
  for (int first=0; first<poses.size(); first+=10) {
    for (int k=0; k<num_lengths; ++k) {
      int last=lastFrameFromSegmentLength(dist,first,lengths[k]);
      if (last<0) continue;
      Matrix delta=Matrix::inv(poses[first])*poses[last];
      std::cout << "S " << first << " " << last << " " << lengths[k];
      for (int row=0; row<3; ++row)
        for (int col=0; col<4; ++col)
          std::cout << " " << delta.val[row][col];
      std::cout << "\n";
    }
  }
  // Same recorded trajectory against itself is only a metric sanity check,
  // not an estimated trajectory or empirical model result.
  auto errors=calcSequenceErrors(poses,poses);
  for (const auto &e:errors)
    std::cout << "E " << e.first_frame << " " << e.len << " "
              << e.t_err << " " << e.r_err << "\n";
  return 0;
}
