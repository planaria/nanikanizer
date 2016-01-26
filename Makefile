CXX ?= clang++

CFLAGS = -std=c++14 -MMD -MP -Wall -Wextra -fopenmp
CFLAGS_DEBUG = -g -O0
CFLAGS_RELEASE = -O3
LDFLAGS = -lpthread

buildtype := release

ifeq ($(buildtype), debug)
	CFLAGS += $(CFLAGS_DEBUG)
else ifeq ($(buildtype), release)
	CFLAGS += $(CFLAGS_RELEASE)
else
	$(error buildtype must be debug or release)
endif

LIBS = 
INCLUDE = -I./include -I./ext/Catch/include

TARGETDIR = ./bin/$(buildtype)
TARGET = $(TARGETDIR)/nanikanizer_tests
SRCDIR = ./tests

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJDIR = ./obj/$(buildtype)
OBJECTS = $(addprefix $(OBJDIR)/, $(notdir $(SOURCES:.cpp=.o)))
DEPENDS = $(OBJECTS:.o=.d)

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJECTS) $(LIBS)
	-mkdir -p $(TARGETDIR)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-mkdir -p $(OBJDIR)
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ -c $<

.PHONY: test
test: $(TARGET)
	$(TARGET)

.PHONY: clean
clean:
	-rm -f $(OBJECTS) $(DEPENDS) $(TARGET)

-include $(DEPENDS)
