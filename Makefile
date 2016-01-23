CC = clang++

CFLAGS = -std=c++14 -MMD -MP -Wall -Wextra
CFLAGS_DEBUG = -g -O0
CFLAGS_RELEASE = -O3
LDFLAGS = -stdlib=libstdc++

buildtype := release
ifeq ($(buildtype), debug)
	CFLAGS += $(CFLAGS_DEBUG)
else ifeq ($(buildtype), release)
	CFLAGS += $(CFLAGS_RELEASE)
else
	$(error buildtype must be debug or release)
endif

LIBS = 
INCLUDE = -I .

TARGETDIR = ./bin/$(buildtype)
TARGET = $(TARGETDIR)/$(shell basename `readlink -f .`)
SRCDIR = ./nanikanizer

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJDIR = ./obj/$(buildtype)
OBJECTS = $(addprefix $(OBJDIR)/, $(notdir $(SOURCES:.cpp=.o)))
DEPENDS = $(OBJECTS:.o=.d)

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJECTS) $(LIBS)
	-mkdir -p $(TARGETDIR)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

.PHONY: clean
clean:
	-rm -f $(OBJECTS) $(DEPENDS) $(TARGET)

-include $(DEPENDS)
