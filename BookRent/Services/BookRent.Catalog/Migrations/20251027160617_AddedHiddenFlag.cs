using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace BookRent.Catalog.Migrations
{
    /// <inheritdoc />
    public partial class AddedHiddenFlag : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "IsVisible",
                table: "Book",
                type: "bit",
                nullable: false,
                defaultValue: false);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "IsVisible",
                table: "Book");
        }
    }
}
