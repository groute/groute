// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#ifndef INTERACTOR_H
#define INTERACTOR_H

#include <string>
#include <iostream>

#include <utils/parser.h>

struct IInteractor
{
    virtual ~IInteractor() { }
    virtual bool RunFirst() = 0;
    virtual bool GetNextCommand(std::string& cmd) = 0;
};

class ConsoleInteractor : public IInteractor
{
    bool RunFirst() override { return true; }
    bool GetNextCommand(std::string& cmd) override
    {
        std::cout << std::endl << "<< ";
        std::getline(std::cin, cmd);
        return !(cmd == "exit" || cmd == "x");
    }
};

class NoInteractor : public IInteractor
{
    bool RunFirst() override { return true; }
    bool GetNextCommand(std::string& cmd) override { return false; }
};

class FileInteractor : public IInteractor
{
    char *line = nullptr;
    size_t lnlen;
    FILE *fpin;

public:
    FileInteractor(const std::string& cmd_file) : lnlen(8192)
    {
        fpin = gk_fopen(cmd_file.c_str(), "r", "Command_File");
        line = (char*)malloc((lnlen)*sizeof(char));
    }

    ~FileInteractor()
    {
        gk_fclose(fpin);   
        free((void *)line); 
    }

    bool RunFirst() override { return false; }
    bool GetNextCommand(std::string& cmd) override
    {
        cmd = std::string(); // reset

        auto n = gk_getline(&line, &lnlen, fpin);
        if (n == -1) return false;
        if (n == 0) return true;

        cmd = std::string(line, line[n-1] == '\n' ? n-1 : n);
        return true;
    }
};

#endif // INTERACTOR_H