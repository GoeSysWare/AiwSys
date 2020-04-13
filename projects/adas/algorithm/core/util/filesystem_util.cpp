#include "filesystem_util.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring> // std::memcpy std::memset

#include <sys/stat.h> // check if folder/file exist

#include <boost/date_time/posix_time/posix_time.hpp>  // boost::make_iterator_range
#include <boost/filesystem.hpp> // boost::filesystem

namespace watrix {
	namespace algorithm {

		// https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
		bool FilesystemUtil::exists(const char* name) 
		{
			struct stat buffer;
			return (stat(name, &buffer) == 0);
		}

		bool FilesystemUtil::not_exists(const char* name) 
		{
			return !FilesystemUtil::exists(name);
		}

		bool FilesystemUtil::mkdir(const char* dir)
		{
			//return boost::filesystem::create_directory(dir);
			return boost::filesystem::create_directories(dir);
		}

		bool FilesystemUtil::rmdir(const char* dir)
		{
			return boost::filesystem::remove_all(dir) > 0;
		}

		bool FilesystemUtil::exists(const std::string& name) 
		{
			struct stat buffer;
			return (stat(name.c_str(), &buffer) == 0);
		}

		bool FilesystemUtil::not_exists(const std::string& name) 
		{
			return !FilesystemUtil::exists(name);
		}

		bool FilesystemUtil::mkdir(const std::string& dir)
		{
			return boost::filesystem::create_directories(dir);
		}

		bool FilesystemUtil::rmdir(const std::string& dir)
		{
			return boost::filesystem::remove_all(dir) > 0;
		}

		/*
		* Get File Name from a Path with or without extension
		*/
		std::string FilesystemUtil::get_filename(
			const std::string& filepath, 
			bool with_extension
		)	
		{
			namespace filesys = boost::filesystem;
			// Create a Path object from File Path
			filesys::path pathObj(filepath);
			
			// Check if file name is required without extension
			if(with_extension == false)
			{
				// Check if file has stem i.e. filename without extension
				if(pathObj.has_stem())
				{
					// return the stem (file name without extension) from path object
					return pathObj.stem().string();
				}
				return "";
			}
			else
			{
				// return the file name with extension from path object
				return pathObj.filename().string();
			}
		}

		void FilesystemUtil::get_lists(
			const std::string& folder, 
			std::vector<std::string>& paths,
			bool reverse
		)
		{
			using namespace boost::filesystem;
			path p(folder);

			if (is_directory(p))
			{
				for (auto& entry : boost::make_iterator_range(directory_iterator(p), {}))
				{
					//std::string ext = entry.path().extension().string();
					
					std::string filepath = folder + entry.path().filename().string();
					paths.push_back(filepath);
					std::cout << "[PATH] "<<filepath << std::endl;
				}

				if (reverse) {
					std::reverse(paths.begin(), paths.end());
				}
			}
		}

		void FilesystemUtil::get_file_lists(
			const std::string& folder, 
			const std::string& format,
			std::vector<std::string>& paths, 
			bool reverse
		)
		{
			using namespace boost::filesystem;
			path p(folder);

			if (is_directory(p))
			{	
				std::vector<double> v_num;
				//std::cout<<" format = "<< format << std::endl;
				for (auto& entry : boost::make_iterator_range(directory_iterator(p), {}))
				{
					std::string ext = entry.path().extension().string();
					//std::cout<<" ext = "<< ext << std::endl;
					if (ext == format) // .jpg
					{	
						std::string file_name = entry.path().filename().string();
						std::string filepath = folder + file_name;
						int end_num = file_name.find_last_of("_");
						double num = std::stod(file_name.substr(0, end_num));
						if (paths.size() == 0){
							paths.push_back(filepath);
							v_num.push_back(num);
						}
						else{
							int idx = 0;
							for ( ; idx<v_num.size(); idx++){
								if (num < v_num[idx]){
									paths.insert(paths.begin()+idx, filepath);
									v_num.insert(v_num.begin()+idx, num);
									break;
								}
							}
							if (idx == v_num.size()){
								paths.push_back(filepath);
								v_num.push_back(num);
							}
						}
						//std::cout << "[PATH] "<<filepath << std::endl;
					}
				}

				if (reverse) {
					std::reverse(paths.begin(), paths.end());
				}
			}
		}

		/*
		void FilesystemUtil::get_image_lists(
			const std::string& folder,
			const std::string& format,
			std::vector<std::string>& paths,
			bool reverse
		)
		{
			return get_file_lists(folder, format, paths, reverse);
		}
		*/

		std::string  FilesystemUtil::str_pad(unsigned int n)
		{
			// 000001,000002
			std::stringstream ss;
			ss << std::setw(6) << std::setfill('0')<< n;
			return ss.str();
		}

		size_t FilesystemUtil::get_stream_size(std::istream *is)
		{
			if (is)
			{
				is->seekg(0, is->end);
				size_t size = (unsigned int)is->tellg();
				is->seekg(0, is->beg);
				return size;
			}
			else {
				return 0;
			}
		}

		size_t FilesystemUtil::get_file_size(const char *filename)
		{
			std::ifstream ifs(filename, std::ifstream::binary);
			size_t size = get_stream_size(&ifs);
			ifs.close();
			return size;
		}

		bool FilesystemUtil::save_stream_to_file(std::istream *blob, size_t size, const char *filename)
		{
			if (blob) {
				std::ofstream outfile(filename, std::ofstream::binary);

				// allocate memory for file content
				char* buffer = new char[size];

				blob->seekg(0, blob->beg);
				// read content of infile
				blob->read((char*)buffer, size);

				// write to outfile
				outfile.write((char*)buffer, size);

				// release dynamically-allocated memory
				delete[] buffer;

				outfile.close();

				return true;
			}
			else {
				return false;
			}
		}

		void FilesystemUtil::new_image_buffer(unsigned char **buffer, unsigned long  *bufferLen, int *width, int *height, int *channel)
		{
			// new default image buffer for later test usage.
			(*width) = 2048;
			(*height) = 1024;
			(*channel) = 1;
			(*bufferLen) = (*width) * (*height) * (*channel);

			unsigned char * pRawBuffer = new unsigned char[(*bufferLen)]; // new buffer
			memset(pRawBuffer, 0, (*bufferLen));

			for (int i = 100; i < 300; i++) // row [height]
			{
				for (int j = 0; j < (*width); j++) // column [width]
				{
					*(pRawBuffer + (*width) * i + j) = (char)255;
				}
			}

			// pass out 
			(*buffer) = pRawBuffer;
		}

		bool FilesystemUtil::save_buffer_to_file(char *buffer, size_t size, const char *filename)
		{
			std::ofstream outfile(filename, std::ofstream::binary);

			if (outfile) {
				// write to outfile
				outfile.write((char*)buffer, size);

				outfile.close();
				return true;
			}
			else {
				return false;
			}

			/*
			FILE *outfile;
			if ((outfile = fopen(filename, "wb")) != nullptr) {
				fwrite(buffer, size, 1, outfile);
				fclose(outfile);
			}
			else
			{
				fprintf(stderr, "can't open %s\n", filename);
				exit(1);
			}
			*/
		}

		boost::shared_ptr<std::vector<char>> FilesystemUtil::get_buffer_from_stream(std::istream *blob, size_t size)
		{
			boost::shared_ptr<std::vector<char>> buffer(new std::vector<char>(size));
			blob->read(&(*buffer.get())[0], size);
			return buffer;
		}

		boost::shared_ptr<std::vector<char>> FilesystemUtil::get_buffer_from_array(char *arr, size_t size)
		{
			boost::shared_ptr<std::vector<char>> buffer(new std::vector<char>(size));
			std::memcpy(&(*buffer.get())[0], arr, size);
			return buffer;
		}

		boost::shared_ptr<std::vector<char>> FilesystemUtil::get_buffer_from_file(const char *filename, size_t &size)
		{
			std::ifstream ifs(filename, std::ifstream::binary);
			size = FilesystemUtil::get_stream_size(&ifs);
			boost::shared_ptr<std::vector<char>> buffer = FilesystemUtil::get_buffer_from_stream(&ifs, size);
			ifs.close();
			return buffer;
		}


	}
}// end namespace