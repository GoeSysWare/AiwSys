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


#include <istream>
#include <string>
#include <vector>

// boost
#include <boost/date_time/posix_time/posix_time.hpp>  
#include <boost/serialization/shared_ptr.hpp> // boost::shared_ptr<T>
using namespace  std;
namespace watrix {
	namespace util {

		class  FilesystemUtil {
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

			static std::string GetFilename(
				const std::string& filepath, 
				bool with_extension = false
			);

			static bool GetFileLists(
				const std::string& folder, 
				std::vector<std::string>& paths,
				bool reverse = false
			);

			static bool GetFileLists(
				const std::string& folder, 
				const std::string& format, // .jpg
				std::vector<std::string>& paths,
				bool reverse = false
			);

			static std::string StrPad(unsigned int n);
		};

	}
}// end namespace