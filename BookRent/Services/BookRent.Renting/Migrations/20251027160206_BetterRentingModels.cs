using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace BookRent.Renting.Migrations
{
    /// <inheritdoc />
    public partial class BetterRentingModels : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "Count",
                table: "BookCounter",
                newName: "MaxCount");

            migrationBuilder.AddColumn<DateTime>(
                name: "ReturnedOn",
                table: "RentedBooks",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "CurrentCount",
                table: "BookCounter",
                type: "int",
                nullable: false,
                defaultValue: 0);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ReturnedOn",
                table: "RentedBooks");

            migrationBuilder.DropColumn(
                name: "CurrentCount",
                table: "BookCounter");

            migrationBuilder.RenameColumn(
                name: "MaxCount",
                table: "BookCounter",
                newName: "Count");
        }
    }
}
