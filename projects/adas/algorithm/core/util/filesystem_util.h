/*
* Software License Agreement
*
*  WATRIX.AI - www.watrix.ai
*  Copyright (c) 2016-2018, Watrix Technology, Inc.
*
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the copyright holder(s) nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  Author: zunlin.ke@watrix.ai (Zunlin Ke)
*
*/
#pragma once
#include "projects/adas/algorithm/algorithm_shared_export.h" // SHARED_EXPORT

#include <istream>
#include <string>
#include <vector>

// boost
#include <boost/date_time/posix_time/posix_time.hpp>  
#include <boost/serialization/shared_ptr.hpp> // boost::shared_ptr<T>

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT FilesystemUtil {
		public:
			// for folder and file
			static bool exists(const char* name);
			static bool not_exists(const char* name);
			static bool mkdir(const char* dir);
			static bool rmdir(const char* dir);

			static bool exists(const std::string& name);
			static bool not_exists(const std::string& name);
			static bool mkdir(const std::string& dir);
			static bool rmdir(const std::string& dir);

			static std::string get_filename(
				const std::string& filepath, 
				bool with_extension = false
			);

			static void get_lists(
				const std::string& folder, 
				std::vector<std::string>& paths,
				bool reverse = false
			);

			static void get_file_lists(
				const std::string& folder, 
				const std::string& format, // .jpg
				std::vector<std::string>& paths,
				bool reverse = false
			);

			/*
			static void get_image_lists(
				const std::string& folder,
				const std::string& format,// .jpg
				std::vector<std::string>& paths,
				bool reverse = false
			);
			*/

			static std::string str_pad(unsigned int n);

			// for stream
			static size_t get_stream_size(std::istream *is);
			static size_t get_file_size(const char *filename);
			static bool save_stream_to_file(std::istream *blob, size_t size, const char *filename);

			static void new_image_buffer(unsigned char **buffer, unsigned long  *bufferLen, int *width, int *height, int *channel);
			static bool save_buffer_to_file(char *buffer, size_t size, const char *filename);

			static boost::shared_ptr<std::vector<char>> get_buffer_from_stream(std::istream *blob, size_t size);
			static boost::shared_ptr<std::vector<char>> get_buffer_from_array(char *arr, size_t size);
			static boost::shared_ptr<std::vector<char>> get_buffer_from_file(const char *filename, size_t &size);
		};

	}
}// end namespace