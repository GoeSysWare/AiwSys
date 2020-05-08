#include "filesystem_util.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring> // std::memcpy std::memset

#include <sys/stat.h> // check if folder/file exist

#include <boost/date_time/posix_time/posix_time.hpp>  // boost::make_iterator_range
#include <boost/filesystem.hpp> // boost::filesystem

namespace watrix {
	namespace util {

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
		std::string FilesystemUtil::GetFilename(
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

		bool FilesystemUtil::GetFileLists(
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
					//WATRIX_INFO << "[PATH] "<<filepath << std::endl;
				}

				if (reverse) {
					std::reverse(paths.begin(), paths.end());
				}
			}
			return paths.size()>0;
		}

		bool FilesystemUtil::GetFileLists(
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
				for (auto& entry : boost::make_iterator_range(directory_iterator(p), {}))
				{
					std::string ext = entry.path().extension().string();
					if (ext == format) // .jpg
					{
						std::string filepath = folder + entry.path().filename().string();
						paths.push_back(filepath);
						//std::cout << "[PATH] "<<filepath << std::endl;
					}
				}

				if (reverse) {
					std::reverse(paths.begin(), paths.end());
				}
			}
			return paths.size()>0;
		}

		std::string  FilesystemUtil::StrPad(unsigned int n)
		{
			// 000001,000002
			std::stringstream ss;
			ss << std::setw(6) << std::setfill('0')<< n;
			return ss.str();
		}

	}
}// end namespace