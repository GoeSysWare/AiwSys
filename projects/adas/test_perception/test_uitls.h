#pragma once

#include <vector>
#include <boost/format.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>


using namespace boost::filesystem;

namespace watrix
{

std::vector<std::vector<std::string>> Test_ParseFiles(std::string filename, std::string dir)
{
    std::vector<std::vector<std::string>> file_list;

    boost::filesystem::fstream pfstream(filename, std::ios_base::in);
    if (!pfstream)
        return file_list;

    while (!pfstream.eof() && pfstream.good())
    {
        char line[2048] = {0};
        std::vector<std::string> string_list;
        pfstream.getline(line, 2048);

        boost::split(string_list, line, boost::is_any_of(",; "));
        if (string_list.size() != 3)
            continue;
        for(auto &name:string_list)
        {
            name = dir + name;
        }
        file_list.push_back(string_list);
    }
    pfstream.close();
    return file_list;
}



}