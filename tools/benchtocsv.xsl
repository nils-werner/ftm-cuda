<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0"
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:output
	method="text" />

<xsl:template match="/">
	<xsl:apply-templates select="benchmark" />
</xsl:template>

<xsl:template match="benchmark">
	<xsl:text>Mode;Matrixmode;Filters;Blocksize;Matrixblocksize;Chunksize;Samples;Turnaround;Roundtrip;Overall
</xsl:text>
	<xsl:apply-templates select="run" />
</xsl:template>

<xsl:template match="run">
<xsl:value-of select="settings/@mode" /><xsl:text>;</xsl:text>
<xsl:value-of select="settings/@matrixmode" /><xsl:text>;</xsl:text>
<xsl:value-of select="settings/@filters" /><xsl:text>;</xsl:text>
<xsl:choose>
	<xsl:when test="settings/@blocksize">
		<xsl:value-of select="settings/@blocksize" /><xsl:text>;</xsl:text>
	</xsl:when>
	<xsl:otherwise>
		<xsl:text>0;</xsl:text>
	</xsl:otherwise>
</xsl:choose>
<xsl:choose>
	<xsl:when test="settings/@matrixblocksize">
		<xsl:value-of select="settings/@matrixblocksize" /><xsl:text>;</xsl:text>
	</xsl:when>
	<xsl:otherwise>
		<xsl:text>0;</xsl:text>
	</xsl:otherwise>
</xsl:choose>
<xsl:value-of select="settings/@chunksize" /><xsl:text>;</xsl:text>
<xsl:choose>
	<xsl:when test="usedsettings/@samples">
		<xsl:value-of select="usedsettings/@samples" /><xsl:text>;</xsl:text>
	</xsl:when>
	<xsl:otherwise>
		<xsl:value-of select="settings/@samples" /><xsl:text>;</xsl:text>
	</xsl:otherwise>
</xsl:choose>
<xsl:choose>
	<xsl:when test="timer">
		<xsl:value-of select="timer[@name = 'turnaround']" /><xsl:text>;</xsl:text>
		<xsl:value-of select="timer[@name = 'roundtrip']" /><xsl:text>;</xsl:text>
		<xsl:value-of select="timer[@name = 'overall']" />
	</xsl:when>
	<xsl:otherwise>
		<xsl:text>0;</xsl:text>
		<xsl:text>0;</xsl:text>
		<xsl:text>0</xsl:text>
	</xsl:otherwise>
</xsl:choose>
<xsl:text>
</xsl:text>
</xsl:template>

</xsl:stylesheet>
