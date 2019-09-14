#include <dlib/dnn.h>
#include <fstream>
#include <iterator>
#include <unordered_map>

template <typename Out>
void split(const std::string &s, char delim, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

template <typename T>
std::ostream &operator<<(std::ostream &o, std::vector<T> data) {
  std::copy(data.cbegin(), data.cend(), std::ostream_iterator<T>(o, " "));
  return o;
}

std::vector<double> getCSV(const std::string &file, char delimiter) {
  std::vector<double> data;
  std::string line, value1, value2;
  std::ifstream f(file, std::ios::in);
  if (!f) {
    std::cerr << "Failed to open file !";
    exit(1);
  }
  std::cout << "reading CSV file " << file << std::endl;
  bool first = true;
  while (getline(f, line)) {
    auto elements = split(line, delimiter);
    if (first) {
      std::cout << elements << std::endl;
      first = false;
    }
  }
  std::cout << "Opened CSV file with " << data.size() << " lines." << std::endl;
  return data;
}

int main(int argc, char *argv[]) {

  // Loads csv file
  getCSV(argv[1], ',');

  // Training data
  //  dlib::matrix<double> x_train(100, 13);

  // std::cout << x_train << std::endl;

  // Defines network
  return 0;
}
