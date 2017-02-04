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
#include <utils/utils.h>



std::map < std::string, graph_t* > g_cachedGraphs;

graph_t* GetCachedGraph(const std::string& graphfile, bool isGR)
{
    auto it = g_cachedGraphs.find(graphfile);
    if (it == g_cachedGraphs.end())
    {
        graph_t *graph;
        if(isGR) 
      graph = ReadGraphGR((char*)graphfile.c_str());
        else
      graph = ReadGraph((char*)graphfile.c_str());

        g_cachedGraphs[graphfile] = graph;
        return graph;
    }
    else
    {
        return it->second;
    }
}

void CleanupGraphs()
{
    for (auto& g : g_cachedGraphs)
    {
        FreeGraph(&g.second);
    }
    g_cachedGraphs.clear();
}

/* stringtoargcargv.cpp -- Parsing a string to std::vector<string>

Copyright (C) 2011 Bernhard Eder

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Bernhard Eder blog_at_bbgen.net

*/

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>

#include <cstdlib>
#include <cstring>

bool _isQuote(char c);
bool _isEscape(char c);
bool _isEscape(char c);
bool _isWhitespace(char c);
std::vector<std::string> parse(const std::string& args);

/*
* Usage:
* int argc;
* char** argv;
* stringToArgcArgv("foo bar", &argc, &argv);
*/
void stringToArgcArgv(const std::string& str, int* argc, char*** argv)
{
    std::vector<std::string> args = parse(str);

    *argv = (char**)std::malloc(args.size() * sizeof(char*));

    int i = 0;
    for (std::vector<std::string>::iterator it = args.begin();
        it != args.end();
        ++it, ++i)
    {
        std::string arg = *it;
        (*argv)[i] = (char*)std::malloc((arg.length() + 1) * sizeof(char));
        std::strcpy((*argv)[i], arg.c_str());
    }

    *argc = args.size();
}

void freeArgcArgv(int* argc, char*** argv)
{
    for (int i = 0; i < *argc; i++)
    {
        free((*argv)[i]);
    }

    free(*argv);
    *argv = NULL;
    *argc = 0;
}

std::vector<std::string> parse(const std::string& args)
{
    std::stringstream ain(args);    // used to iterate over input string
    ain >> std::noskipws;           // do not skip white spaces
    std::vector<std::string> oargs; // output list of arguments

    std::stringstream currentArg("");
    currentArg >> std::noskipws;

    // current state
    enum State {
        InArg,      // currently scanning an argument
        InArgQuote, // currently scanning an argument (which started with quotes)
        OutOfArg    // currently not scanning an argument
    };
    State currentState = OutOfArg;

    char currentQuoteChar = '\0'; // to distinguish between ' and " quotations
    // this allows to use "foo'bar"

    char c;
    while (!ain.eof() && (ain >> c)) { // iterate char by char

        if (_isQuote(c)) {
            switch (currentState) {
            case OutOfArg:
                currentArg.str(std::string());
            case InArg:
                currentState = InArgQuote;
                currentQuoteChar = c;
                break;

            case InArgQuote:
                if (c == currentQuoteChar)
                    currentState = InArg;
                else
                    currentArg << c;
                break;
            }

        }
        else if (_isWhitespace(c)) {
            switch (currentState) {
            case InArg:
                oargs.push_back(currentArg.str());
                currentState = OutOfArg;
                break;
            case InArgQuote:
                currentArg << c;
                break;
            case OutOfArg:
                // nothing
                break;
            }
        }
        else if (_isEscape(c)) {
            switch (currentState) {
            case OutOfArg:
                currentArg.str(std::string());
                currentState = InArg;
            case InArg:
            case InArgQuote:
                if (ain.eof())
                {
#ifdef WIN32
                    // Windows doesn't care about an escape character at the end.
                    // It just adds \ to the arg.
                    currentArg << c;
#else
                    throw(std::runtime_error("Found Escape Character at end of file."));
#endif
                }
                else
                {
#ifdef WIN32
                    // Windows only escapes the " character.
                    // Every other character is just printed and the \ is added itself.
                    char c1 = c;
                    ain >> c;
                    if (c != '\"')
                        currentArg << c1; // only ignore \ when next char is "
                    ain.unget();
#else
                    ain >> c;
                    currentArg << c;
#endif
                }
                break;
            }
        }
        else {
            switch (currentState) {
            case InArg:
            case InArgQuote:
                currentArg << c;
                break;

            case OutOfArg:
                currentArg.str(std::string());
                currentArg << c;
                currentState = InArg;
                break;
            }
        }
    }

    if (currentState == InArg)
        oargs.push_back(currentArg.str());
    else if (currentState == InArgQuote)
        throw(std::runtime_error("Starting quote has no ending quote."));

    return oargs;
}

bool _isQuote(char c)
{
    if (c == '\"')
        return true;
    else if (c == '\'')
        return true;

    return false;
}

bool _isEscape(char c)
{
    if (c == '\\')
        return true;

    return false;
}

bool _isWhitespace(char c)
{
    if (c == ' ')
        return true;
    else if (c == '\t')
        return true;

    return false;
}
