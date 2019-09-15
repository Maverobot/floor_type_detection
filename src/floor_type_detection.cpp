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

std::vector<dlib::matrix<double>> getCSV(const std::string &file,
                                         char delimiter = ',') {
  std::vector<dlib::matrix<double>> data;
  std::string line, value1, value2;
  std::ifstream f(file, std::ios::in);
  if (!f) {
    std::cerr << "Failed to open file !";
    exit(1);
  }
  std::cout << "reading CSV file " << file << std::endl;
  bool first = true;
  while (getline(f, line)) {
    if (first) {
      first = false;
      continue;
    }
    auto elements = split(line, delimiter);
    dlib::matrix<double> line_data(1, elements.size());
    dlib::set_all_elements(line_data, 0);
    std::for_each(elements.cbegin(), elements.cend(),
                  [i = 0, &line_data](auto &e) mutable {
                    line_data(0, i) = stod(e);
                    std::cout << e << " to " << line_data(i) << std::endl;
                  });
    std::cout << std::endl;
    std::cout << line_data << std::endl << std::endl;
    data.push_back(line_data);
  }
  std::cout << "Opened CSV file with " << data.size() << " lines." << std::endl;
  return data;
}

int main(int argc, char *argv[]) {
  getCSV(argv[1]);
  return 0;
}
